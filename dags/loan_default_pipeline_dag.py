"""
loan_default_pipeline_dag.py

End-to-end Airflow DAG for CS611 Assignment 2.

Pipeline stages (each bronze/silver source is its own task for visibility
and independent failure isolation):

  BRONZE  : bronze_lms, bronze_clickstream, bronze_attributes, bronze_financials
  SILVER  : silver_lms, silver_clickstream, silver_attributes, silver_financials
  GOLD    : gold_label_store, gold_feature_store
  ML      : train_model -> run_inference -> run_monitoring

Dependency structure:
  - Each bronze_X feeds its corresponding silver_X
  - gold_label_store depends only on silver_lms
  - gold_feature_store depends on silver_attributes, silver_financials, silver_clickstream
  - train_model depends on BOTH gold tables
  - run_inference depends on train_model
  - run_monitoring depends on run_inference

This DAG is for MANUAL TRIGGER (no schedule) and supports backfill — each
task internally loops over the full date range (2023-01-01 to 2024-12-01),
so one DAG run processes the entire history.
"""

import sys
import os
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator

sys.path.insert(0, "/opt/airflow")
sys.path.insert(0, "/opt/airflow/utils")


# ─────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────
START_DATE_STR = "2023-01-01"
END_DATE_STR   = "2024-12-01"

BRONZE_LMS_DIR          = "/opt/airflow/datamart/bronze/lms/"
BRONZE_CLICKSTREAM_DIR  = "/opt/airflow/datamart/bronze/clickstream/"
BRONZE_ATTRIBUTES_DIR   = "/opt/airflow/datamart/bronze/attributes/"
BRONZE_FINANCIALS_DIR   = "/opt/airflow/datamart/bronze/financials/"

SILVER_LOAN_DAILY_DIR   = "/opt/airflow/datamart/silver/loan_daily/"
SILVER_CLICKSTREAM_DIR  = "/opt/airflow/datamart/silver/clickstream_daily/"
SILVER_ATTRIBUTES_DIR   = "/opt/airflow/datamart/silver/attributes_daily/"
SILVER_FINANCIALS_DIR   = "/opt/airflow/datamart/silver/financials_daily/"

GOLD_LABEL_STORE_DIR    = "/opt/airflow/datamart/gold/label_store/"
GOLD_FEATURE_STORE_DIR  = "/opt/airflow/datamart/gold/feature_store/"


def _make_dirs(*paths):
    for p in paths:
        os.makedirs(p, exist_ok=True)


def _generate_month_list(start_date_str, end_date_str):
    start = datetime.strptime(start_date_str, "%Y-%m-%d")
    end = datetime.strptime(end_date_str, "%Y-%m-%d")
    dates = []
    current = datetime(start.year, start.month, 1)
    while current <= end:
        dates.append(current.strftime("%Y-%m-%d"))
        if current.month == 12:
            current = datetime(current.year + 1, 1, 1)
        else:
            current = datetime(current.year, current.month + 1, 1)
    return dates


def _get_spark(app_name):
    import pyspark
    spark = pyspark.sql.SparkSession.builder.appName(app_name).master("local[*]").getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    return spark


# ─────────────────────────────────────────────────────────────────────────
# BRONZE tasks (one per source)
# ─────────────────────────────────────────────────────────────────────────
def task_bronze_lms(**context):
    import utils.data_processing_bronze_table as bronze_mod
    spark = _get_spark("airflow_bronze_lms")
    _make_dirs(BRONZE_LMS_DIR)
    date_list = _generate_month_list(START_DATE_STR, END_DATE_STR)
    for date_str in date_list:
        bronze_mod.process_bronze_table(date_str, BRONZE_LMS_DIR, spark)
    spark.stop()
    print(f"[bronze_lms] complete: {len(date_list)} months")


def task_bronze_clickstream(**context):
    import utils.data_processing_bronze_table as bronze_mod
    spark = _get_spark("airflow_bronze_clickstream")
    _make_dirs(BRONZE_CLICKSTREAM_DIR)
    date_list = _generate_month_list(START_DATE_STR, END_DATE_STR)
    for date_str in date_list:
        bronze_mod.process_bronze_clickstream(date_str, BRONZE_CLICKSTREAM_DIR, spark)
    spark.stop()
    print(f"[bronze_clickstream] complete: {len(date_list)} months")


def task_bronze_attributes(**context):
    import utils.data_processing_bronze_table as bronze_mod
    spark = _get_spark("airflow_bronze_attributes")
    _make_dirs(BRONZE_ATTRIBUTES_DIR)
    date_list = _generate_month_list(START_DATE_STR, END_DATE_STR)
    for date_str in date_list:
        bronze_mod.process_bronze_attributes(date_str, BRONZE_ATTRIBUTES_DIR, spark)
    spark.stop()
    print(f"[bronze_attributes] complete: {len(date_list)} months")


def task_bronze_financials(**context):
    import utils.data_processing_bronze_table as bronze_mod
    spark = _get_spark("airflow_bronze_financials")
    _make_dirs(BRONZE_FINANCIALS_DIR)
    date_list = _generate_month_list(START_DATE_STR, END_DATE_STR)
    for date_str in date_list:
        bronze_mod.process_bronze_financials(date_str, BRONZE_FINANCIALS_DIR, spark)
    spark.stop()
    print(f"[bronze_financials] complete: {len(date_list)} months")


# ─────────────────────────────────────────────────────────────────────────
# SILVER tasks (one per source)
# ─────────────────────────────────────────────────────────────────────────
def task_silver_lms(**context):
    import utils.data_processing_silver_table as silver_mod
    spark = _get_spark("airflow_silver_lms")
    _make_dirs(SILVER_LOAN_DAILY_DIR)
    date_list = _generate_month_list(START_DATE_STR, END_DATE_STR)
    for date_str in date_list:
        silver_mod.process_silver_table(date_str, BRONZE_LMS_DIR, SILVER_LOAN_DAILY_DIR, spark)
    spark.stop()
    print(f"[silver_lms] complete: {len(date_list)} months")


def task_silver_clickstream(**context):
    import utils.data_processing_silver_table as silver_mod
    spark = _get_spark("airflow_silver_clickstream")
    _make_dirs(SILVER_CLICKSTREAM_DIR)
    date_list = _generate_month_list(START_DATE_STR, END_DATE_STR)
    for date_str in date_list:
        silver_mod.process_silver_clickstream(date_str, BRONZE_CLICKSTREAM_DIR, SILVER_CLICKSTREAM_DIR, spark)
    spark.stop()
    print(f"[silver_clickstream] complete: {len(date_list)} months")


def task_silver_attributes(**context):
    import utils.data_processing_silver_table as silver_mod
    spark = _get_spark("airflow_silver_attributes")
    _make_dirs(SILVER_ATTRIBUTES_DIR)
    date_list = _generate_month_list(START_DATE_STR, END_DATE_STR)
    for date_str in date_list:
        silver_mod.process_silver_attributes(date_str, BRONZE_ATTRIBUTES_DIR, SILVER_ATTRIBUTES_DIR, spark)
    spark.stop()
    print(f"[silver_attributes] complete: {len(date_list)} months")


def task_silver_financials(**context):
    import utils.data_processing_silver_table as silver_mod
    spark = _get_spark("airflow_silver_financials")
    _make_dirs(SILVER_FINANCIALS_DIR)
    date_list = _generate_month_list(START_DATE_STR, END_DATE_STR)
    for date_str in date_list:
        silver_mod.process_silver_financials(date_str, BRONZE_FINANCIALS_DIR, SILVER_FINANCIALS_DIR, spark)
    spark.stop()
    print(f"[silver_financials] complete: {len(date_list)} months")


# ─────────────────────────────────────────────────────────────────────────
# GOLD tasks (label store + feature store)
# ─────────────────────────────────────────────────────────────────────────
def task_gold_label_store(**context):
    import utils.data_processing_gold_table as gold_mod
    spark = _get_spark("airflow_gold_labels")
    _make_dirs(GOLD_LABEL_STORE_DIR)
    date_list = _generate_month_list(START_DATE_STR, END_DATE_STR)
    for date_str in date_list:
        gold_mod.process_labels_gold_table(
            date_str, SILVER_LOAN_DAILY_DIR, GOLD_LABEL_STORE_DIR, spark, dpd=30, mob=6
        )
    spark.stop()
    print(f"[gold_label_store] complete: {len(date_list)} months")


def task_gold_feature_store(**context):
    import utils.data_processing_gold_table as gold_mod
    spark = _get_spark("airflow_gold_features")
    _make_dirs(GOLD_FEATURE_STORE_DIR)
    date_list = _generate_month_list(START_DATE_STR, END_DATE_STR)
    for date_str in date_list:
        gold_mod.process_feature_store_gold_table(
            date_str,
            SILVER_ATTRIBUTES_DIR,
            SILVER_FINANCIALS_DIR,
            SILVER_CLICKSTREAM_DIR,
            GOLD_FEATURE_STORE_DIR,
            spark
        )
    spark.stop()
    print(f"[gold_feature_store] complete: {len(date_list)} months")


# ─────────────────────────────────────────────────────────────────────────
# ML tasks
# ─────────────────────────────────────────────────────────────────────────
def task_train_model(**context):
    os.chdir("/opt/airflow")
    from utils.model_training import train_and_select_best_model
    artefact = train_and_select_best_model()
    print(f"[train_model] best model: {artefact['model_name']}")
    print(f"[train_model] test metrics: {artefact['test_metrics']}")


def task_run_inference(**context):
    os.chdir("/opt/airflow")
    from utils.model_inference import run_inference_for_range
    predictions = run_inference_for_range(START_DATE_STR, END_DATE_STR)
    print(f"[run_inference] total predictions: {len(predictions) if predictions is not None else 0}")


def task_run_monitoring(**context):
    os.chdir("/opt/airflow")
    from utils.model_monitoring import run_monitoring, plot_monitoring_results
    monitoring_df = run_monitoring(START_DATE_STR, END_DATE_STR)
    plot_monitoring_results(monitoring_df)
    print("[run_monitoring] complete.")
    print(monitoring_df[["snapshot_date", "predicted_default_rate", "psi", "psi_flag", "auc"]].to_string())


# ─────────────────────────────────────────────────────────────────────────
# DAG definition
# ─────────────────────────────────────────────────────────────────────────
default_args = {
    "owner": "data_scientist",
    "retries": 1,
    "retry_delay": timedelta(minutes=2),
}

with DAG(
    dag_id="loan_default_pipeline",
    default_args=default_args,
    description="End-to-end loan default ML pipeline: bronze -> silver -> gold -> train -> infer -> monitor",
    schedule_interval=None,
    start_date=datetime(2023, 1, 1),
    catchup=False,
    tags=["cs611", "assignment2", "loan_default"],
) as dag:

    # ── Bronze ──
    bronze_lms          = PythonOperator(task_id="bronze_lms", python_callable=task_bronze_lms)
    bronze_clickstream  = PythonOperator(task_id="bronze_clickstream", python_callable=task_bronze_clickstream)
    bronze_attributes   = PythonOperator(task_id="bronze_attributes", python_callable=task_bronze_attributes)
    bronze_financials   = PythonOperator(task_id="bronze_financials", python_callable=task_bronze_financials)

    # ── Silver ──
    silver_lms          = PythonOperator(task_id="silver_lms", python_callable=task_silver_lms)
    silver_clickstream  = PythonOperator(task_id="silver_clickstream", python_callable=task_silver_clickstream)
    silver_attributes   = PythonOperator(task_id="silver_attributes", python_callable=task_silver_attributes)
    silver_financials   = PythonOperator(task_id="silver_financials", python_callable=task_silver_financials)

    # ── Gold ──
    gold_label_store    = PythonOperator(task_id="gold_label_store", python_callable=task_gold_label_store)
    gold_feature_store  = PythonOperator(task_id="gold_feature_store", python_callable=task_gold_feature_store)

    # ── ML ──
    train_model      = PythonOperator(task_id="train_model", python_callable=task_train_model)
    run_inference    = PythonOperator(task_id="run_inference", python_callable=task_run_inference)
    run_monitoring   = PythonOperator(task_id="run_monitoring", python_callable=task_run_monitoring)

    # ── Dependency wiring ──
    bronze_lms >> silver_lms
    bronze_clickstream >> silver_clickstream
    bronze_attributes >> silver_attributes
    bronze_financials >> silver_financials

    silver_lms >> gold_label_store

    [silver_attributes, silver_financials, silver_clickstream] >> gold_feature_store

    [gold_label_store, gold_feature_store] >> train_model

    train_model >> run_inference >> run_monitoring
