"""Метрики качества модели на отложенном месяце."""

import numpy as np
import pandas as pd

THRESHOLD = 0.5


def classification_metrics(y_true: pd.Series, proba: np.ndarray) -> dict:
    """Считает PR-AUC, ROC-AUC, F1, precision и recall при пороге 0.5.

    PR-AUC (average precision) — основная метрика: доля фрода меньше 1%,
    и ROC-AUC на таком дисбалансе почти не различает модели.
    Если в выборке один класс, ранговые метрики возвращаются как None.
    """
    from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score

    pred = (proba > THRESHOLD).astype(int)
    both_classes = pd.Series(y_true).nunique() > 1
    return {
        "pr_auc": float(average_precision_score(y_true, proba)) if both_classes else None,
        "roc_auc": float(roc_auc_score(y_true, proba)) if both_classes else None,
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
    }
