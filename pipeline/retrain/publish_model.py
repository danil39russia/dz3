"""Публикация или отклонение модели и фиксация нового окна обучения."""

import json
import logging
import os
from datetime import date
from pathlib import Path

from pipeline.db import advance_training_state, get_conn
from pipeline.settings import CURRENT_DATASET, MODELS_DIR, REJECTED_DIR

logger = logging.getLogger(__name__)


def _model_report(eval_info: dict, status: str) -> str:
    keys = ["version", "batch_month", "train_months", "test_month", "train_rows", "test_rows",
            "metrics", "champion_version", "champion_metrics", "decision_reason"]
    return json.dumps({**{k: eval_info[k] for k in keys}, "status": status}, ensure_ascii=False, indent=2)


def _move_staging(eval_info: dict, target: Path) -> None:
    """Переносит черновик в target. Если перенос уже был (повтор задачи) — ничего не делает."""
    staging = Path(eval_info["staging_model"])
    if staging.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        os.replace(staging, target)
    elif not target.exists():
        raise FileNotFoundError(f"Нет файла модели: {staging}")


def publish_model(eval_info: dict) -> str:
    """Публикует новую модель в общий каталог, откуда читает сервис скоринга.

    Читает: черновик .staging_<version>.cbm.
    Пишет: my_catboost_<version>.cbm, my_catboost_<version>.metrics.json и
    атомарно (через os.replace) latest.txt — после этого сервис подхватывает модель
    на следующем сообщении без перезапуска.
    Повтор: если черновик уже переименован, файл модели не трогается,
    latest.txt и метрики просто перезаписываются теми же значениями.
    """
    version = eval_info["version"]
    _move_staging(eval_info, MODELS_DIR / f"my_catboost_{version}.cbm")
    (MODELS_DIR / f"my_catboost_{version}.metrics.json").write_text(
        _model_report(eval_info, "published"), encoding="utf-8")

    latest_tmp = MODELS_DIR / f".latest.{version}.txt"
    latest_tmp.write_text(version + "\n", encoding="utf-8")
    os.replace(latest_tmp, MODELS_DIR / "latest.txt")
    logger.info("Опубликована модель %s (%s)", version, eval_info["decision_reason"])
    return version


def reject_model(eval_info: dict) -> str:
    """Сохраняет отклонённую модель отдельно от опубликованных.

    Читает: черновик .staging_<version>.cbm.
    Пишет: rejected/my_catboost_<version>.cbm и .metrics.json с причиной отказа.
    latest.txt не меняется — сервис продолжает скорить текущей моделью.
    Повтор: если черновик уже перенесён, задача только перезаписывает метрики.
    """
    version = eval_info["version"]
    _move_staging(eval_info, REJECTED_DIR / f"my_catboost_{version}.cbm")
    (REJECTED_DIR / f"my_catboost_{version}.metrics.json").write_text(
        _model_report(eval_info, "rejected"), encoding="utf-8")
    logger.info("Модель %s не раскатывается: %s", version, eval_info["decision_reason"])
    return version


def commit_training_window(eval_info: dict) -> dict:
    """Фиксирует сдвиг окна обучения после того, как модель сохранена.

    Запускается после publish_model или reject_model (одна из веток пропущена).
    Читает: pending_window.parquet. Пишет: current.parquet и training_state.
    Окно сдвигается в обоих случаях: месяц уже использован для обучения и проверки,
    отказ в раскатке не должен блокировать следующие месяцы.
    Повтор: сначала проверяется, что файл модели действительно сохранён.
    pending-файл переносится только если он есть, а training_state
    выставляется в абсолютные значения от batch_month — повторный запуск
    не сдвигает окно второй раз.
    """
    version = eval_info["version"]
    saved = [MODELS_DIR / f"my_catboost_{version}.cbm", REJECTED_DIR / f"my_catboost_{version}.cbm"]
    if not any(p.exists() for p in saved):
        raise FileNotFoundError(f"Модель {version} не сохранена, окно не сдвигаем")

    pending_window = Path(eval_info["dataset_path"])
    if pending_window.exists():
        os.replace(pending_window, CURRENT_DATASET)
        logger.info("Окно обучения обновлено: %s", CURRENT_DATASET)
    elif not CURRENT_DATASET.exists():
        raise FileNotFoundError(f"Нет нового окна обучения: {pending_window}")
    else:
        logger.info("Окно уже перенесено в %s, повторно не сдвигаем.", CURRENT_DATASET)

    conn = get_conn()
    try:
        advance_training_state(conn, date.fromisoformat(eval_info["batch_month"]))
    finally:
        conn.close()
    return {"version": version, "batch_month": eval_info["batch_month"], "deployed": eval_info["deploy"]}
