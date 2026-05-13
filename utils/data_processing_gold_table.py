import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import pprint
import pyspark
import pyspark.sql.functions as F
import argparse

from pyspark.sql.functions import col
from pyspark.sql.types import StringType, IntegerType, FloatType, DateType


def process_labels_gold_table(snapshot_date_str, silver_loan_daily_directory, gold_label_store_directory, spark, dpd, mob):
    
    # prepare arguments
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")
    
    # connect to silver table
    partition_name = "silver_loan_daily_" + snapshot_date_str.replace('-','_') + '.parquet'
    filepath = silver_loan_daily_directory + partition_name
    df = spark.read.parquet(filepath)
    print('loaded from:', filepath, 'row count:', df.count())

    # get customer at mob
    df = df.filter(col("mob") == mob)

    # get label
    df = df.withColumn("label", F.when(col("dpd") >= dpd, 1).otherwise(0).cast(IntegerType()))
    df = df.withColumn("label_def", F.lit(str(dpd)+'dpd_'+str(mob)+'mob').cast(StringType()))

    # select columns to save
    df = df.select("loan_id", "Customer_ID", "label", "label_def", "snapshot_date")

    # save gold table - IRL connect to database to write
    partition_name = "gold_label_store_" + snapshot_date_str.replace('-','_') + '.parquet'
    filepath = gold_label_store_directory + partition_name
    df.write.mode("overwrite").parquet(filepath)
    # df.toPandas().to_parquet(filepath,
    #           compression='gzip')
    print('saved to:', filepath)
    
    return df

# ─────────────────────────────────────────────
# Feature store gold
def process_feature_store_gold_table(
    snapshot_date_str,
    silver_attributes_directory,
    silver_financials_directory,
    silver_clickstream_directory,
    gold_feature_store_directory,
    spark
):
    """
    Build feature store gold table — ML-ready input features (X).
 
    Design:
    - One row per customer per snapshot_date
    - Left join: attributes (base) -> financials -> clickstream
      Using left join means customers with missing data in one source are
      still included, with nulls filled by 0 / 'Unknown'
    - NO data from LMS loan repayment table to avoid target leakage
    - snapshot_date represents the point-in-time of the feature snapshot
      (i.e. what we knew about the customer at loan application time)
    """
    # ── Load silver tables ──────────────────────────────────────────────
    attr_partition = "silver_attributes_" + snapshot_date_str.replace('-', '_') + '.parquet'
    attr_df = spark.read.parquet(silver_attributes_directory + attr_partition)
 
    fin_partition = "silver_financials_" + snapshot_date_str.replace('-', '_') + '.parquet'
    fin_df = spark.read.parquet(silver_financials_directory + fin_partition)
 
    click_partition = "silver_clickstream_" + snapshot_date_str.replace('-', '_') + '.parquet'
    click_df = spark.read.parquet(silver_clickstream_directory + click_partition)
 
    print(f'{snapshot_date_str} | attributes: {attr_df.count()} | financials: {fin_df.count()} | clickstream: {click_df.count()}')
 
    # ── Rename snapshot_date in financials and clickstream to avoid collision on join ──
    fin_df = fin_df.drop("snapshot_date")
    click_df = click_df.drop("snapshot_date")
 
    # ── Join: attributes is the base (left), join financials and clickstream ──
    # This ensures we keep all customers even if they are missing in one source
    df = attr_df \
        .join(fin_df, on="Customer_ID", how="left") \
        .join(click_df, on="Customer_ID", how="left")
 
    # ── Fill nulls introduced by left joins ──
    # Numeric clickstream features: 0 means no activity
    fe_columns = [f"fe_{i}" for i in range(1, 21)]
    df = df.fillna(0, subset=fe_columns)
 
    # Numeric financial columns: fill with 0
    numeric_financial_cols = [
        "Annual_Income", "Monthly_Inhand_Salary", "Num_Bank_Accounts",
        "Num_Credit_Card", "Interest_Rate", "Num_of_Loan",
        "Delay_from_due_date", "Num_of_Delayed_Payment", "Changed_Credit_Limit",
        "Num_Credit_Inquiries", "Outstanding_Debt", "Credit_Utilization_Ratio",
        "Total_EMI_per_month", "Amount_invested_monthly", "Monthly_Balance",
        "Credit_History_Months"
    ]
    df = df.fillna(0.0, subset=numeric_financial_cols)
 
    # Categorical financial columns: fill with 'Unknown'
    categorical_financial_cols = [
        "Credit_Mix", "Type_of_Loan", "Payment_of_Min_Amount", "Payment_Behaviour"
    ]
    df = df.fillna("Unknown", subset=categorical_financial_cols)
 
    # ── Final schema enforcement ──
    df = df.withColumn("snapshot_date", col("snapshot_date").cast(DateType()))
    df = df.withColumn("Customer_ID", col("Customer_ID").cast(StringType()))
 
    print(f'{snapshot_date_str} | feature store row count: {df.count()}')
 
    # ── Save gold feature store ──
    partition_name = "gold_feature_store_" + snapshot_date_str.replace('-', '_') + '.parquet'
    filepath = gold_feature_store_directory + partition_name
    df.write.mode("overwrite").parquet(filepath)
    print('saved to:', filepath)
 
    return df