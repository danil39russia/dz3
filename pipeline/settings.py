"""Общие пути артефактов и параметры обучения, поставки батчей и проверок качества."""

import os
from pathlib import Path

TRAINING_DIR = Path(os.getenv("TRAINING_DATA_DIR", "/opt/airflow/training_data"))
MODELS_DIR = Path(os.getenv("MODELS_DIR", "/opt/airflow/models"))
REJECTED_DIR = MODELS_DIR / "rejected"
DQ_REPORTS_DIR = Path(os.getenv("DQ_REPORTS_DIR", "/opt/airflow/dq_reports"))
FRAUD_SRC = os.getenv("FRAUD_SRC", "/opt/airflow/fraud_src")

CURRENT_DATASET = TRAINING_DIR / "current.parquet"
PENDING_WINDOW = TRAINING_DIR / "pending_window.parquet"
BATCHES_DIR = TRAINING_DIR / "batches"
SPLITS_DIR = TRAINING_DIR / "splits"
SUPPLY_BATCH = TRAINING_DIR / "supply_batch.parquet"
DQ_SAMPLE = TRAINING_DIR / "dq_sample.parquet"

# Должен совпадать с CAT_FEATURES в fraud_detector/src/preprocessing.py
CAT_FEATURES = ["merch", "cat_id", "gender", "one_city", "us_state", "jobs", "cat_daypart"]
LABEL = "target"

# Окно обучения: WINDOW_MONTHS месяцев, из них последний — test, остальные — train
WINDOW_MONTHS = 4

# Поставка размеченных данных: размер батча и глубина хранения витрины
SAMPLE_SIZE = 50_000
HISTORY_YEARS = 2

# Раскатка модели: основная метрика и допустимая просадка относительно текущей модели
PRIMARY_METRIC = "pr_auc"
MIN_IMPROVEMENT = 0.0
