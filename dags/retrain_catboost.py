"""DAG скользящего переобучения CatBoost с проверкой улучшения модели.

Файл только связывает задачи. Логика каждой задачи лежит в pipeline.retrain.
"""

from datetime import datetime, timedelta

from airflow.decorators import dag, task

from pipeline.retrain.compare_with_champion import choose_deploy_branch, evaluate_against_champion
from pipeline.retrain.extract_month_batch import extract_month_batch
from pipeline.retrain.fit_catboost import fit_catboost
from pipeline.retrain.preprocess_dataset import preprocess_dataset
from pipeline.retrain.publish_model import commit_training_window, publish_model, reject_model
from pipeline.retrain.slide_training_window import slide_training_window
from pipeline.retrain.time_train_test_split import time_train_test_split


@dag(
    dag_id="retrain_catboost",
    description="Раз в 5 минут сдвигает 4-месячное окно, переобучает CatBoost и раскатывает, если модель не хуже",
    schedule="*/5 * * * *",
    start_date=datetime(2026, 1, 1),
    catchup=False,
    max_active_runs=1,
    dagrun_timeout=timedelta(minutes=15),
    default_args={"retries": 1, "retry_delay": timedelta(seconds=30)},
    tags=["mlops", "catboost"],
    doc_md="""
### Переобучение my_catboost

1. Из витрины `labeled_transactions` берётся следующий календарный месяц.
2. Месяц добавляется в окно из 4 месяцев, самый старый месяц выбрасывается.
3. Train — три более ранних месяца, test — самый свежий.
4. Препроцессинг — `run_preproc` из `fraud_detector/src` (тот же, что в онлайн-скоринге).
5. CatBoost обучается и сохраняется черновиком.
6. Текущая опубликованная модель оценивается на том же test-месяце.
7. Если PR-AUC новой модели не ниже — `publish_model`, иначе `reject_model`.
8. `commit_training_window` сдвигает указатель месяца только после сохранения модели.
    """,
)
def retrain_catboost():
    """Связывает извлечение месяца, сдвиг окна, split, препроцессинг, обучение, сравнение и раскатку."""
    extracted = task(extract_month_batch, task_id="extract_month_batch")()
    slid = task(slide_training_window, task_id="slide_training_window")(extracted)
    split = task(time_train_test_split, task_id="time_train_test_split")(slid)
    processed = task(preprocess_dataset, task_id="preprocess_dataset")(split)
    trained = task(fit_catboost, task_id="fit_catboost")(processed)
    evaluated = task(evaluate_against_champion, task_id="evaluate_against_champion")(trained)

    branch = task.branch(choose_deploy_branch, task_id="choose_deploy")(evaluated)
    published = task(publish_model, task_id="publish_model")(evaluated)
    rejected = task(reject_model, task_id="reject_model")(evaluated)
    committed = task(
        commit_training_window,
        task_id="commit_training_window",
        trigger_rule="none_failed_min_one_success",
    )(evaluated)

    branch >> [published, rejected]
    [published, rejected] >> committed


retrain_catboost()
