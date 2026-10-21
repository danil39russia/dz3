"""Обучение стартовой (базовой) модели, которая зашита в образ fraud_detector.

Базовая модель намеренно простая: обучена только на январе 2019 и 100 итерациях.
Она скорит поток до первой публикации из DAG и служит первым «чемпионом»,
с которым сравниваются переобученные модели.

Запуск из корня репозитория (нужны catboost, pandas, pyarrow):
    python fraud_detector/tools/train_initial_model.py data/train.csv
"""
import sys
from pathlib import Path

import pandas as pd
from catboost import CatBoostClassifier

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from preprocessing import CAT_FEATURES, run_preproc  # noqa: E402


def main(csv_path: str) -> None:
    raw = pd.read_csv(csv_path, parse_dates=["transaction_time"])
    raw = raw[(raw["transaction_time"] >= "2019-01-01") & (raw["transaction_time"] < "2019-02-01")]
    data = run_preproc(raw)
    x, y = data.drop(columns=["target"]), data["target"]

    model = CatBoostClassifier(iterations=100, depth=4, learning_rate=0.1, random_seed=42,
                               allow_writing_files=False, verbose=25)
    model.fit(x, y, cat_features=CAT_FEATURES)
    out = ROOT / "models" / "my_catboost.cbm"
    model.save_model(str(out))
    print(f"Сохранено: {out}, признаков: {len(model.feature_names_)}, строк: {len(x)}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "data/train.csv")
