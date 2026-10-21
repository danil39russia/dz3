"""Обучение CatBoost на скользящем окне."""

import logging
from datetime import datetime, timezone

import pandas as pd

from pipeline.retrain.metrics import classification_metrics
from pipeline.settings import CAT_FEATURES, LABEL, MODELS_DIR

logger = logging.getLogger(__name__)

EVAL_SHARE = 0.15


def prepare_features(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Отделяет target от признаков и приводит категориальные колонки к строкам.

    Категории совпадают с теми, что ожидает онлайн-скоринг после run_preproc.
    Возвращает матрицу признаков и вектор целевой переменной.
    """
    if LABEL not in frame.columns:
        raise ValueError(f"В датасете нет колонки {LABEL}")
    features = frame.drop(columns=[LABEL]).copy()
    for column in CAT_FEATURES:
        if column not in features.columns:
            raise ValueError(f"После препроцессинга нет категориального признака {column}")
        features[column] = features[column].astype(str)
    return features, frame[LABEL]


def fit_catboost(preproc_info: dict) -> dict:
    """Обучает CatBoost на train-месяцах окна и считает метрики на test-месяце.

    Читает: обработанные train/test parquet из preproc_info (train упорядочен по времени).
    Пишет: черновик модели .staging_<version>.cbm в каталоге моделей. latest.txt
    не трогает, поэтому сервис скоринга черновик не видит.
    Повтор: каждый запуск пишет черновик с новой версией, старые черновики ни на что
    не влияют, состояние окна и опубликованные модели не меняются.

    Для ранней остановки используется хвост train (последние 15% по времени), а не
    test-месяц: test должен оставаться честной отложенной выборкой для сравнения
    с текущей моделью.
    """
    from catboost import CatBoostClassifier

    train = pd.read_parquet(preproc_info["train_proc_path"])
    test = pd.read_parquet(preproc_info["test_proc_path"])
    x_train, y_train = prepare_features(train)
    x_test, y_test = prepare_features(test)

    split_at = int(len(x_train) * (1 - EVAL_SHARE))
    model = CatBoostClassifier(
        iterations=300,
        depth=6,
        learning_rate=0.08,
        loss_function="Logloss",
        eval_metric="PRAUC",
        early_stopping_rounds=40,
        random_seed=42,
        allow_writing_files=False,
        thread_count=2,
        verbose=50,
    )
    model.fit(
        x_train.iloc[:split_at],
        y_train.iloc[:split_at],
        cat_features=CAT_FEATURES,
        eval_set=(x_train.iloc[split_at:], y_train.iloc[split_at:]),
        use_best_model=True,
    )

    metrics = classification_metrics(y_test, model.predict_proba(x_test)[:, 1])
    logger.info("Новая модель на test %s: %s", preproc_info["test_month"], metrics)

    version = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    staging_model = MODELS_DIR / f".staging_{version}.cbm"
    model.save_model(str(staging_model))
    logger.info("Черновик модели сохранён: %s", staging_model)
    return {
        "version": version,
        "staging_model": str(staging_model),
        "test_proc_path": preproc_info["test_proc_path"],
        "dataset_path": preproc_info["dataset_path"],
        "batch_month": preproc_info["batch_month"],
        "train_months": preproc_info["train_months"],
        "test_month": preproc_info["test_month"],
        "train_rows": int(len(x_train)),
        "test_rows": int(len(x_test)),
        "metrics": metrics,
    }
