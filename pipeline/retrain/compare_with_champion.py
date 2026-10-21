"""Сравнение новой модели (challenger) с опубликованной (champion)."""

import logging

import pandas as pd

from pipeline.retrain.fit_catboost import prepare_features
from pipeline.retrain.metrics import classification_metrics
from pipeline.settings import MIN_IMPROVEMENT, MODELS_DIR, PRIMARY_METRIC

logger = logging.getLogger(__name__)


def read_champion_version() -> str | None:
    """Версия из latest.txt — модель, которой сейчас скорит сервис."""
    latest = MODELS_DIR / "latest.txt"
    if not latest.exists():
        return None
    return latest.read_text(encoding="utf-8").strip() or None


def evaluate_against_champion(train_info: dict) -> dict:
    """Считает метрики текущей опубликованной модели на том же test-месяце.

    Читает: latest.txt и файл модели-чемпиона из каталога моделей, обработанный
    test-месяц из train_info. Ничего не пишет.
    Обе модели не видели test-месяц при обучении, поэтому сравнение честное:
    чемпион обучен на окне, которое заканчивается предыдущим месяцем.
    Решение: публикуем, если PRIMARY_METRIC новой модели не ниже, чем у чемпиона,
    с допуском MIN_IMPROVEMENT. Если чемпиона нет или его нельзя применить
    к текущим признакам, новая модель публикуется.
    Повтор: задача только читает файлы и детерминированно даёт то же решение.
    """
    from catboost import CatBoostClassifier

    new_value = train_info["metrics"][PRIMARY_METRIC]
    champion_version = read_champion_version()
    champion_metrics = None

    if champion_version:
        path = MODELS_DIR / f"my_catboost_{champion_version}.cbm"
        try:
            champion = CatBoostClassifier()
            champion.load_model(str(path))
            x_test, y_test = prepare_features(pd.read_parquet(train_info["test_proc_path"]))
            proba = champion.predict_proba(x_test[list(champion.feature_names_)])[:, 1]
            champion_metrics = classification_metrics(y_test, proba)
        except Exception as exc:  # noqa: BLE001 — несовместимая или битая модель не должна блокировать раскатку
            logger.warning("Не удалось оценить чемпиона %s: %s", champion_version, exc)

    old_value = champion_metrics[PRIMARY_METRIC] if champion_metrics else None
    if old_value is None or new_value is None:
        deploy = old_value is None
        reason = "нет сравнимой метрики чемпиона" if deploy else "у новой модели метрика не определена"
    else:
        deploy = new_value >= old_value - MIN_IMPROVEMENT
        reason = f"{PRIMARY_METRIC}: новая {new_value:.4f} vs текущая {old_value:.4f}"

    logger.info("Чемпион %s, решение: %s (%s)", champion_version, "публикуем" if deploy else "отклоняем", reason)
    return {
        **train_info,
        "champion_version": champion_version,
        "champion_metrics": champion_metrics,
        "deploy": deploy,
        "decision_reason": reason,
    }


def choose_deploy_branch(eval_info: dict) -> str:
    """Ветвление DAG: publish_model, если новая модель не хуже, иначе reject_model."""
    return "publish_model" if eval_info["deploy"] else "reject_model"
