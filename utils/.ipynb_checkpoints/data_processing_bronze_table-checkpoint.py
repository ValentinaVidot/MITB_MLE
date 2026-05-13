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


def process_bronze_table(snapshot_date_str, bronze_lms_directory, spark):
    # prepare arguments
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")
    
    # connect to source back end - IRL connect to back end source system
    csv_file_path = "data/lms_loan_daily.csv"

    # load data - IRL ingest from back end source system
    df = spark.read.csv(csv_file_path, header=True, inferSchema=True).filter(col('snapshot_date') == snapshot_date)
    print(snapshot_date_str + 'row count:', df.count())
    
    # save bronze table to datamart - IRL connect to database to write
    partition_name = "bronze_loan_daily_" + snapshot_date_str.replace('-','_') + '.csv'
    filepath = bronze_lms_directory + partition_name
    df.toPandas().to_csv(filepath, index=False)
    print('saved to:', filepath)

    return df

# Clickstream bronze
# ─────────────────────────────────────────────
def process_bronze_clickstream(snapshot_date_str, bronze_clickstream_directory, spark):
    """
    Ingest raw clickstream CSV and save as bronze partition.
    Bronze = raw data, no transformation, just filter by snapshot_date and save.
    """
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")
 
    csv_file_path = "data/feature_clickstream.csv"
    df = spark.read.csv(csv_file_path, header=True, inferSchema=True) \
               .filter(col('snapshot_date') == snapshot_date)
    print(snapshot_date_str, 'clickstream bronze row count:', df.count())
 
    partition_name = "bronze_clickstream_" + snapshot_date_str.replace('-', '_') + '.parquet'
    filepath = bronze_clickstream_directory + partition_name
    df.write.mode("overwrite").parquet(filepath)
    print('saved to:', filepath)
 
    return df
 
 
# ─────────────────────────────────────────────
# Attributes bronze
def process_bronze_attributes(snapshot_date_str, bronze_attributes_directory, spark):
    """
    Ingest raw customer attributes CSV and save as bronze partition.
    Bronze = raw data including PII (Name, SSN) — PII removal happens at silver.
    """
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")
 
    csv_file_path = "data/features_attributes.csv"
    df = spark.read.csv(csv_file_path, header=True, inferSchema=True) \
               .filter(col('snapshot_date') == snapshot_date)
    print(snapshot_date_str, 'attributes bronze row count:', df.count())
 
    partition_name = "bronze_attributes_" + snapshot_date_str.replace('-', '_') + '.parquet'
    filepath = bronze_attributes_directory + partition_name
    df.write.mode("overwrite").parquet(filepath)
    print('saved to:', filepath)
 
    return df
 
 
# ─────────────────────────────────────────────
#  Financials bronze
def process_bronze_financials(snapshot_date_str, bronze_financials_directory, spark):
    """
    Ingest raw customer financials CSV and save as bronze partition.
    Bronze = raw data, dirty values preserved — cleaning happens at silver.
    """
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")
 
    csv_file_path = "data/features_financials.csv"
    df = spark.read.csv(csv_file_path, header=True, inferSchema=True) \
               .filter(col('snapshot_date') == snapshot_date)
    print(snapshot_date_str, 'financials bronze row count:', df.count())
 
    partition_name = "bronze_financials_" + snapshot_date_str.replace('-', '_') + '.parquet'
    filepath = bronze_financials_directory + partition_name
    df.write.mode("overwrite").parquet(filepath)
    print('saved to:', filepath)
 
    return df