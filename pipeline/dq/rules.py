"""Правила качества данных для размеченных транзакций.

Каждое правило получает DataFrame и месяц проверки и возвращает
(passed, значение метрики, порог в читаемом виде).
"""

import pandas as pd

from pipeline.db import LABELED_COLUMNS

KEY_COLUMNS = ["transaction_time", "amount", "cat_id", "us_state", "lat", "lon",
               "merchant_lat", "merchant_lon", "target"]
KNOWN_CATEGORIES = {
    "entertainment", "food_dining", "gas_transport", "grocery_net", "grocery_pos",
    "health_fitness", "home", "kids_pets", "misc_net", "misc_pos", "personal_care",
    "shopping_net", "shopping_pos", "travel",
}
MAX_NULL_SHARE = 0.01
MAX_AMOUNT = 50_000
MIN_ROWS = 10_000
FRAUD_RATE_RANGE = (0.001, 0.03)
MAX_DUPLICATE_SHARE = 0.01


def schema(df, month):
    missing = [c for c in LABELED_COLUMNS if c not in df.columns]
    return not missing, f"нет колонок: {missing}" if missing else "все 18 колонок", "все колонки витрины"


def completeness(df, month):
    share = df[KEY_COLUMNS].isna().mean()
    worst = share.idxmax()
    return share.max() <= MAX_NULL_SHARE, f"{worst}: {share.max():.2%}", f"пропусков ≤ {MAX_NULL_SHARE:.0%}"


def target_domain(df, month):
    bad = (~df["target"].isin([0, 1])).sum()
    return bad == 0, f"{bad} строк вне {{0, 1}}", "target ∈ {0, 1}"


def amount_range(df, month):
    bad = ((df["amount"] <= 0) | (df["amount"] > MAX_AMOUNT)).sum()
    return bad == 0, f"{bad} строк, max {df['amount'].max():.2f}", f"0 < amount ≤ {MAX_AMOUNT}"


def coordinates(df, month):
    bad = (~df["lat"].between(-90, 90) | ~df["lon"].between(-180, 180)
           | ~df["merchant_lat"].between(-90, 90) | ~df["merchant_lon"].between(-180, 180)).sum()
    return bad == 0, f"{bad} строк", "широта ∈ [-90, 90], долгота ∈ [-180, 180]"


def categorical_domains(df, month):
    bad_gender = (~df["gender"].isin(["M", "F"])).sum()
    bad_state = (~df["us_state"].astype(str).str.fullmatch(r"[A-Z]{2}")).sum()
    unknown_cat = sorted(set(df["cat_id"].dropna()) - KNOWN_CATEGORIES)
    ok = bad_gender == 0 and bad_state == 0 and not unknown_cat
    return ok, f"gender: {bad_gender}, us_state: {bad_state}, новые cat_id: {unknown_cat}", \
        "gender ∈ {M, F}, us_state — 2 буквы, cat_id из справочника 14 категорий"


def time_in_month(df, month):
    ts = pd.to_datetime(df["transaction_time"])
    start = pd.Timestamp(month)
    bad = (ts.isna() | (ts < start) | (ts >= start + pd.DateOffset(months=1))).sum()
    return bad == 0, f"{bad} строк вне месяца", f"transaction_time в пределах {start:%Y-%m}"


def volume(df, month):
    return len(df) >= MIN_ROWS, f"{len(df)} строк", f"≥ {MIN_ROWS} строк в месяце"


def fraud_rate(df, month):
    rate = df["target"].mean()
    lo, hi = FRAUD_RATE_RANGE
    return lo <= rate <= hi, f"{rate:.3%}", f"доля фрода ∈ [{lo:.1%}, {hi:.0%}]"


def duplicates(df, month):
    share = df.duplicated().mean()
    return share <= MAX_DUPLICATE_SHARE, f"{share:.3%}", f"дубликатов ≤ {MAX_DUPLICATE_SHARE:.0%}"


# (id, название, функция, критичность)
RULES = [
    ("DQ1", "Схема витрины", schema, "critical"),
    ("DQ2", "Полнота ключевых полей", completeness, "critical"),
    ("DQ3", "Допустимые значения target", target_domain, "critical"),
    ("DQ4", "Диапазон суммы транзакции", amount_range, "critical"),
    ("DQ5", "Корректность координат", coordinates, "critical"),
    ("DQ6", "Справочные значения категорий", categorical_domains, "critical"),
    ("DQ7", "Время внутри проверяемого месяца", time_in_month, "critical"),
    ("DQ8", "Объём месяца", volume, "warning"),
    ("DQ9", "Доля мошенничества", fraud_rate, "warning"),
    ("DQ10", "Дубликаты строк", duplicates, "warning"),
]
