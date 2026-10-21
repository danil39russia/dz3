"""Препроцессинг транзакций для обучения и онлайн-скоринга.

Все признаки считаются построчно и не зависят от статистик обучающей выборки,
поэтому одна и та же функция run_preproc используется и в DAG переобучения,
и в сервисе скоринга для одиночного сообщения из Kafka.
"""
import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

EARTH_RADIUS_KM = 6371.0

# Категориальные признаки после препроцессинга (их же ждёт CatBoost)
CAT_FEATURES = ["merch", "cat_id", "gender", "one_city", "us_state", "jobs", "cat_daypart"]

# Колонки, которые заменяются производными признаками или не несут сигнала
DROP_COLUMNS = [
    "transaction_time", "name_1", "name_2", "street", "post_code",
    "lat", "lon", "merchant_lat", "merchant_lon", "population_city",
]


def distance_km(lat1, lon1, lat2, lon2):
    """Расстояние по большому кругу между клиентом и мерчантом, км."""
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    a = (np.sin((lat2 - lat1) / 2) ** 2
         + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2)
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def daypart(hour: pd.Series) -> pd.Series:
    """Часть суток: ночь (22–5), утро (6–11), день (12–17), вечер (18–21)."""
    bins = np.select(
        [(hour >= 22) | (hour < 6), hour < 12, hour < 18],
        ["night", "morning", "day"],
        default="evening",
    )
    return pd.Series(bins, index=hour.index)


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    ts = pd.to_datetime(out["transaction_time"])

    # время: циклическое кодирование часа, день недели, ночные и выходные операции
    hour = ts.dt.hour
    out["hour"] = hour
    out["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    out["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    out["day_of_week"] = ts.dt.dayofweek
    out["is_weekend"] = (out["day_of_week"] >= 5).astype(int)
    out["is_night"] = ((hour >= 22) | (hour < 6)).astype(int)

    # сумма: лог-масштаб, «круглые» суммы, крупные операции
    amount = out["amount"].astype(float)
    out["log_amount"] = np.log1p(amount)
    out["is_round_amount"] = (np.isclose(amount % 1, 0)).astype(int)
    out["is_large_amount"] = (amount > 500).astype(int)

    # география: расстояние до мерчанта и размер города
    out["distance_km"] = distance_km(out["lat"], out["lon"], out["merchant_lat"], out["merchant_lon"])
    out["log_city_pop"] = np.log1p(out["population_city"].astype(float))
    out["amount_per_log_pop"] = amount / (out["log_city_pop"] + 1)

    # категории: префикс "fraud_" у всех мерчантов не несёт информации
    out["merch"] = out["merch"].astype(str).str.replace("^fraud_", "", regex=True)
    out["cat_daypart"] = out["cat_id"].astype(str) + "_" + daypart(hour)
    for column in CAT_FEATURES:
        out[column] = out[column].fillna("unknown").astype(str)

    return out.drop(columns=[c for c in DROP_COLUMNS if c in out.columns])


def run_preproc(input_df: pd.DataFrame) -> pd.DataFrame:
    """Сырые транзакции -> признаки модели. Колонка target, если есть, сохраняется."""
    output_df = add_features(input_df)
    logger.debug("Preprocessing done. Output shape: %s", output_df.shape)
    return output_df
