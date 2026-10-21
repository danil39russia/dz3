"""DAG проверки качества данных размеченной витрины.

Файл только связывает задачи. Логика и правила лежат в pipeline.dq.
"""

from datetime import datetime, timedelta

from airflow.decorators import dag, task

from pipeline.dq.checks import fail_on_critical, publish_dq_report, run_dq_checks
from pipeline.dq.extract_dq_sample import extract_dq_sample


@dag(
    dag_id="data_quality_check",
    description="Раз в 5 минут проверяет свежий месяц витрины и публикует HTML-отчёт",
    schedule="*/5 * * * *",
    start_date=datetime(2026, 1, 1),
    catchup=False,
    max_active_runs=1,
    dagrun_timeout=timedelta(minutes=10),
    default_args={"retries": 1, "retry_delay": timedelta(seconds=30)},
    tags=["mlops", "data-quality"],
    doc_md="""
### Проверка качества данных

1. Выгружается самый свежий месяц `labeled_transactions`.
2. К нему применяются 10 правил (см. README).
3. HTML-отчёт публикуется в `/opt/airflow/dq_reports` (на хосте — `./dq_reports`).
4. Если не прошло критичное правило, прогон помечается упавшим.
    """,
)
def data_quality_check():
    """Связывает выгрузку месяца, проверки, публикацию отчёта и итоговый статус."""
    sample = task(extract_dq_sample, task_id="extract_dq_sample")()
    results = task(run_dq_checks, task_id="run_dq_checks")(sample)
    report = task(publish_dq_report, task_id="publish_dq_report")(results)
    task(fail_on_critical, task_id="fail_on_critical")(report)


data_quality_check()
