"""Прогон правил качества данных и публикация HTML-отчёта."""

import html
import json
import logging
import os
from datetime import datetime, timezone

import pandas as pd

from pipeline.dq.rules import RULES
from pipeline.settings import DQ_REPORTS_DIR

logger = logging.getLogger(__name__)


def run_dq_checks(sample_info: dict) -> dict:
    """Применяет правила из pipeline.dq.rules к выгруженному месяцу.

    Читает: dq_sample.parquet. Ничего не пишет.
    Падение одного правила (исключение) не прерывает остальные: оно
    записывается как непройденное с текстом ошибки.
    Повтор: чистая функция от файла выгрузки, результат тот же.
    """
    df = pd.read_parquet(sample_info["sample_path"])
    results = []
    for rule_id, name, check, severity in RULES:
        try:
            passed, value, expectation = check(df, sample_info["month"])
        except Exception as exc:  # noqa: BLE001
            passed, value, expectation = False, f"ошибка: {exc}", "-"
        results.append({"id": rule_id, "name": name, "severity": severity,
                        "passed": bool(passed), "value": str(value), "expectation": expectation})
        logger.info("%s %s: %s (%s)", rule_id, name, "OK" if passed else "FAIL", value)
    return {"month": sample_info["month"], "rows": int(len(df)), "results": results}


def _render(report: dict, generated_at: str) -> str:
    failed_critical = sum(not r["passed"] and r["severity"] == "critical" for r in report["results"])
    failed_warning = sum(not r["passed"] and r["severity"] == "warning" for r in report["results"])
    status = "FAILED" if failed_critical else ("WARNING" if failed_warning else "PASSED")
    color = {"PASSED": "#1a7f37", "WARNING": "#9a6700", "FAILED": "#cf222e"}[status]
    rows = "\n".join(
        f"<tr class='{'ok' if r['passed'] else r['severity']}'><td>{r['id']}</td><td>{html.escape(r['name'])}</td>"
        f"<td>{r['severity']}</td><td>{html.escape(r['expectation'])}</td><td>{html.escape(r['value'])}</td>"
        f"<td>{'PASS' if r['passed'] else 'FAIL'}</td></tr>"
        for r in report["results"]
    )
    return f"""<!doctype html>
<html lang="ru"><head><meta charset="utf-8"><title>DQ report {report['month']}</title>
<style>
body{{font-family:system-ui,sans-serif;margin:32px;color:#1f2328}}
table{{border-collapse:collapse;width:100%}}td,th{{border:1px solid #d0d7de;padding:6px 10px;text-align:left}}
th{{background:#f6f8fa}}tr.ok td:last-child{{color:#1a7f37;font-weight:600}}
tr.critical td:last-child{{color:#cf222e;font-weight:600}}tr.warning td:last-child{{color:#9a6700;font-weight:600}}
.status{{color:{color};font-weight:700}}
</style></head><body>
<h1>Data Quality: labeled_transactions</h1>
<p>Проверяемый месяц: <b>{report['month'][:7]}</b> · строк: <b>{report['rows']}</b> · сформирован: {generated_at} UTC</p>
<p>Статус: <span class="status">{status}</span> (critical: {failed_critical}, warning: {failed_warning})</p>
<table><tr><th>ID</th><th>Правило</th><th>Критичность</th><th>Требование</th><th>Факт</th><th>Результат</th></tr>
{rows}
</table></body></html>"""


def publish_dq_report(report: dict) -> dict:
    """Пишет HTML-отчёт и JSON с результатами в каталог отчётов рядом с каталогом моделей.

    Читает: результаты run_dq_checks.
    Пишет: dq_report_<YYYYMM>_<время>.html, тот же отчёт как latest.html и .json.
    latest.html заменяется атомарно через os.replace.
    Повтор: создаёт новый файл с новой меткой времени, прошлые отчёты не меняет.
    """
    DQ_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc)
    stamp = now.strftime("%Y%m%dT%H%M%S")
    name = f"dq_report_{report['month'][:7].replace('-', '')}_{stamp}"
    page = _render(report, now.strftime("%Y-%m-%d %H:%M:%S"))

    (DQ_REPORTS_DIR / f"{name}.html").write_text(page, encoding="utf-8")
    (DQ_REPORTS_DIR / f"{name}.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp = DQ_REPORTS_DIR / ".latest.html"
    tmp.write_text(page, encoding="utf-8")
    os.replace(tmp, DQ_REPORTS_DIR / "latest.html")
    logger.info("Отчёт качества данных: %s", DQ_REPORTS_DIR / f"{name}.html")
    return {**report, "report": f"{name}.html"}


def fail_on_critical(report: dict) -> None:
    """Помечает прогон DAG упавшим, если не прошло хотя бы одно критичное правило.

    Запускается после публикации отчёта, поэтому отчёт доступен и при ошибке.
    Ничего не читает и не пишет, повтор безопасен.
    """
    failed = [r["id"] for r in report["results"] if not r["passed"] and r["severity"] == "critical"]
    if failed:
        raise ValueError(f"Не пройдены критичные правила качества: {failed}. Отчёт: {report['report']}")
    logger.info("Критичные правила пройдены. Отчёт: %s", report["report"])
