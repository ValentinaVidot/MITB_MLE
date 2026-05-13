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


def process_silver_table(snapshot_date_str, bronze_lms_directory, silver_loan_daily_directory, spark):
    # prepare arguments
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")
    
    # connect to bronze table
    partition_name = "bronze_loan_daily_" + snapshot_date_str.replace('-','_') + '.csv'
    filepath = bronze_lms_directory + partition_name
    df = spark.read.csv(filepath, header=True, inferSchema=True)
    print('loaded from:', filepath, 'row count:', df.count())

    # clean data: enforce schema / data type
    # Dictionary specifying columns and their desired datatypes
    column_type_map = {
        "loan_id": StringType(),
        "Customer_ID": StringType(),
        "loan_start_date": DateType(),
        "tenure": IntegerType(),
        "installment_num": IntegerType(),
        "loan_amt": FloatType(),
        "due_amt": FloatType(),
        "paid_amt": FloatType(),
        "overdue_amt": FloatType(),
        "balance": FloatType(),
        "snapshot_date": DateType(),
    }

    for column, new_type in column_type_map.items():
        df = df.withColumn(column, col(column).cast(new_type))

    # augment data: add month on book
    df = df.withColumn("mob", col("installment_num").cast(IntegerType()))

    # augment data: add days past due
    df = df.withColumn("installments_missed", F.ceil(col("overdue_amt") / col("due_amt")).cast(IntegerType())).fillna(0)
    df = df.withColumn("first_missed_date", F.when(col("installments_missed") > 0, F.add_months(col("snapshot_date"), -1 * col("installments_missed"))).cast(DateType()))
    df = df.withColumn("dpd", F.when(col("overdue_amt") > 0.0, F.datediff(col("snapshot_date"), col("first_missed_date"))).otherwise(0).cast(IntegerType()))

    # save silver table - IRL connect to database to write
    partition_name = "silver_loan_daily_" + snapshot_date_str.replace('-','_') + '.parquet'
    filepath = silver_loan_daily_directory + partition_name
    df.write.mode("overwrite").parquet(filepath)
    # df.toPandas().to_parquet(filepath,
    #           compression='gzip')
    print('saved to:', filepath)
    
    return df

# ─────────────────────────────────────────────
# Clickstream silver
def process_silver_clickstream(snapshot_date_str, bronze_clickstream_directory, silver_clickstream_directory, spark):
    """
    Clean clickstream bronze data:
    - Cast all fe_1..fe_20 columns to IntegerType
    - Fill any nulls with 0 (no activity = 0)
    """
    partition_name = "bronze_clickstream_" + snapshot_date_str.replace('-', '_') + '.parquet'
    filepath = bronze_clickstream_directory + partition_name
    df = spark.read.parquet(filepath)
    print('loaded from:', filepath, 'row count:', df.count())
 
    # Cast Customer_ID and snapshot_date
    df = df.withColumn("Customer_ID", col("Customer_ID").cast(StringType()))
    df = df.withColumn("snapshot_date", col("snapshot_date").cast(DateType()))
 
    # Cast all fe_* feature columns to IntegerType and fill nulls with 0
    fe_columns = [f"fe_{i}" for i in range(1, 21)]
    for fe_col in fe_columns:
        df = df.withColumn(fe_col, col(fe_col).cast(IntegerType()))
    df = df.fillna(0, subset=fe_columns)
 
    partition_name = "silver_clickstream_" + snapshot_date_str.replace('-', '_') + '.parquet'
    filepath = silver_clickstream_directory + partition_name
    df.write.mode("overwrite").parquet(filepath)
    print('saved to:', filepath)
 
    return df
 
 
# ─────────────────────────────────────────────
# Attributes silver
def process_silver_attributes(snapshot_date_str, bronze_attributes_directory, silver_attributes_directory, spark):
    """
    Clean attributes bronze data:
    - Drop PII columns: Name, SSN
    - Cast Age to IntegerType; clamp to valid range [18, 100] — data has outliers like -500 and 925
    - Cast Occupation to StringType; fill nulls with 'Unknown'
    """
    partition_name = "bronze_attributes_" + snapshot_date_str.replace('-', '_') + '.parquet'
    filepath = bronze_attributes_directory + partition_name
    df = spark.read.parquet(filepath)
    print('loaded from:', filepath, 'row count:', df.count())
 
    # Drop PII — Name and SSN are not useful for ML and are a privacy risk
    df = df.drop("Name", "SSN")
 
    # Enforce schema
    df = df.withColumn("Customer_ID", col("Customer_ID").cast(StringType()))
    df = df.withColumn("snapshot_date", col("snapshot_date").cast(DateType()))
 
    # Clean Age: cast to int, null out impossible values, then fill with median proxy (35)
    df = df.withColumn("Age", col("Age").cast(IntegerType()))
    df = df.withColumn("Age", F.when((col("Age") >= 18) & (col("Age") <= 100), col("Age")).otherwise(None))
    df = df.fillna({"Age": 35})  # fill with reasonable default
 
    # Clean Occupation
    df = df.withColumn("Occupation", col("Occupation").cast(StringType()))
    df = df.fillna({"Occupation": "Unknown"})
 
    partition_name = "silver_attributes_" + snapshot_date_str.replace('-', '_') + '.parquet'
    filepath = silver_attributes_directory + partition_name
    df.write.mode("overwrite").parquet(filepath)
    print('saved to:', filepath)
 
    return df
 
 
# ─────────────────────────────────────────────
# Financials silver
def process_silver_financials(snapshot_date_str, bronze_financials_directory, silver_financials_directory, spark):
    """
    Clean financials bronze data:
    - Strip trailing underscores from Annual_Income (dirty source data e.g. '52312.68_')
    - Replace '_' sentinel values in Credit_Mix with null, then fill with 'Unknown'
    - Fill null Type_of_Loan with 'Unknown'
    - Parse Credit_History_Age from string ('10 Years and 9 Months') to integer months
    - Cast all numeric columns to correct types
    """
    partition_name = "bronze_financials_" + snapshot_date_str.replace('-', '_') + '.parquet'
    filepath = bronze_financials_directory + partition_name
    df = spark.read.parquet(filepath)
    print('loaded from:', filepath, 'row count:', df.count())
 
    df = df.withColumn("Customer_ID", col("Customer_ID").cast(StringType()))
    df = df.withColumn("snapshot_date", col("snapshot_date").cast(DateType()))
 
    # Fix Annual_Income: strip trailing underscores/spaces then cast to Float
    df = df.withColumn("Annual_Income", regexp_replace(trim(col("Annual_Income").cast(StringType())), "_", "").cast(FloatType()))
 
    # Fix Monthly_Inhand_Salary
    df = df.withColumn("Monthly_Inhand_Salary", col("Monthly_Inhand_Salary").cast(FloatType()))
 
    # Fix Credit_Mix: treat '_' as unknown
    df = df.withColumn("Credit_Mix",
        F.when(trim(col("Credit_Mix").cast(StringType())) == "_", "Unknown")
         .otherwise(col("Credit_Mix").cast(StringType())))
    df = df.fillna({"Credit_Mix": "Unknown"})
 
    # Fix Type_of_Loan: fill nulls
    df = df.withColumn("Type_of_Loan", col("Type_of_Loan").cast(StringType()))
    df = df.fillna({"Type_of_Loan": "Unknown"})
 
    # Parse Credit_History_Age: '10 Years and 9 Months' -> integer months (total)
    # Extract years and months separately then combine
    df = df.withColumn("credit_history_age_str", col("Credit_History_Age").cast(StringType()))
    df = df.withColumn("cha_years",
        F.regexp_extract(col("credit_history_age_str"), r"(\d+)\s+Year", 1).cast(IntegerType()))
    df = df.withColumn("cha_months",
        F.regexp_extract(col("credit_history_age_str"), r"(\d+)\s+Month", 1).cast(IntegerType()))
    df = df.withColumn("Credit_History_Months",
        (F.coalesce(col("cha_years"), F.lit(0)) * 12 + F.coalesce(col("cha_months"), F.lit(0))).cast(IntegerType()))
    df = df.drop("Credit_History_Age", "credit_history_age_str", "cha_years", "cha_months")
 
    # Cast remaining numeric columns
    numeric_cols = {
        "Num_Bank_Accounts": IntegerType(),
        "Num_Credit_Card": IntegerType(),
        "Interest_Rate": FloatType(),
        "Num_of_Loan": IntegerType(),
        "Delay_from_due_date": IntegerType(),
        "Num_of_Delayed_Payment": IntegerType(),
        "Changed_Credit_Limit": FloatType(),
        "Num_Credit_Inquiries": FloatType(),
        "Outstanding_Debt": FloatType(),
        "Credit_Utilization_Ratio": FloatType(),
        "Payment_of_Min_Amount": StringType(),   # Yes/No categorical
        "Total_EMI_per_month": FloatType(),
        "Amount_invested_monthly": FloatType(),
        "Payment_Behaviour": StringType(),        # categorical
        "Monthly_Balance": FloatType(),
    }
    for c, t in numeric_cols.items():
        df = df.withColumn(c, col(c).cast(t))
 
    # Fill remaining numeric nulls with 0
    numeric_fill_cols = [c for c, t in numeric_cols.items() if t != StringType()]
    df = df.fillna(0.0, subset=numeric_fill_cols)
 
    partition_name = "silver_financials_" + snapshot_date_str.replace('-', '_') + '.parquet'
    filepath = silver_financials_directory + partition_name
    df.write.mode("overwrite").parquet(filepath)
    print('saved to:', filepath)
 
    return df