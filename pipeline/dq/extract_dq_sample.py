"""Выгрузка месяца витрины, который проверяется на качество данных."""

import logging

import pandas as pd

from pipeline.db import LABELED_COLUMNS, get_conn, month_end_exclusive
from pipeline.settings import DQ_SAMPLE, TRAINING_DIR

logger = logging.getLogger(__name__)


def extract_dq_sample() -> dict:
    """Выгружает самый свежий месяц labeled_transactions — то, что последним попало в витрину.

    Читает: витрину labeled_transactions (только SELECT).
    Пишет: dq_sample.parquet в каталоге обучающих данных.
    Повтор: файл перезаписывается целиком, витрина не меняется.
    Возвращает путь к выгрузке и проверяемый месяц.
    """
    TRAINING_DIR.mkdir(parents=True, exist_ok=True)
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT date_trunc('month', max(transaction_time))::date FROM labeled_transactions")
            month = cur.fetchone()[0]
        if month is None:
            raise RuntimeError("Витрина labeled_transactions пуста.")
        frame = pd.read_sql(
            f"""
            SELECT {", ".join(LABELED_COLUMNS)}
            FROM labeled_transactions
            WHERE transaction_time >= %s AND transaction_time < %s
            """,
            conn,
            params=(month, month_end_exclusive(month)),
        )
    finally:
        conn.close()

    frame.to_parquet(DQ_SAMPLE, index=False)
    logger.info("Для проверки качества выгружен месяц %s: %s строк", month, len(frame))
    return {"sample_path": str(DQ_SAMPLE), "month": month.isoformat()}
