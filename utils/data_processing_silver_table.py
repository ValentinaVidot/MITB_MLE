import os
from datetime import datetime
import pyspark.sql.functions as F
from pyspark.sql.functions import col, regexp_replace, trim
from pyspark.sql.types import StringType, IntegerType, FloatType, DateType


# ─────────────────────────────────────────────
# LMS loan daily silver
# ─────────────────────────────────────────────
def process_silver_table(snapshot_date_str, bronze_lms_directory, silver_loan_daily_directory, spark):
    """Clean and enrich LMS bronze data: enforce schema, compute MOB and DPD."""
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")

    partition_name = "bronze_loan_daily_" + snapshot_date_str.replace('-', '_') + '.csv'
    filepath = bronze_lms_directory + partition_name
    df = spark.read.csv(filepath, header=True, inferSchema=True)
    print('loaded from:', filepath, 'row count:', df.count())

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

    df = df.withColumn("mob", col("installment_num").cast(IntegerType()))

    # FIX: guard against divide-by-zero when due_amt is 0 (Spark's ANSI mode
    # throws a hard error on bare division by zero rather than returning NULL).
    # F.when(...due_amt > 0...) ensures we never divide by zero; rows where
    # due_amt is 0 simply get 0 installments missed.
    df = df.withColumn(
        "installments_missed",
        F.when(col("due_amt") > 0, F.ceil(col("overdue_amt") / col("due_amt")))
         .otherwise(0)
         .cast(IntegerType())
    ).fillna(0, subset=["installments_missed"])

    df = df.withColumn("first_missed_date", F.when(col("installments_missed") > 0, F.add_months(col("snapshot_date"), -1 * col("installments_missed"))).cast(DateType()))
    df = df.withColumn("dpd", F.when(col("overdue_amt") > 0.0, F.datediff(col("snapshot_date"), col("first_missed_date"))).otherwise(0).cast(IntegerType()))

    partition_name = "silver_loan_daily_" + snapshot_date_str.replace('-', '_') + '.parquet'
    filepath = silver_loan_daily_directory + partition_name
    df.write.mode("overwrite").parquet(filepath)
    print('saved to:', filepath)

    return df


# ─────────────────────────────────────────────
# Clickstream silver
# ─────────────────────────────────────────────
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

    df = df.withColumn("Customer_ID", col("Customer_ID").cast(StringType()))
    df = df.withColumn("snapshot_date", col("snapshot_date").cast(DateType()))

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
# ─────────────────────────────────────────────
def process_silver_attributes(snapshot_date_str, bronze_attributes_directory, silver_attributes_directory, spark):
    """
    Clean attributes bronze data:
    - Drop PII columns: Name, SSN
    - Cast Age to IntegerType; clamp to valid range [18, 100] — data has outliers
      like -500, 925, AND dirty strings like '41_' with a trailing underscore.
    - Cast Occupation to StringType; fill nulls with 'Unknown'
    """
    partition_name = "bronze_attributes_" + snapshot_date_str.replace('-', '_') + '.parquet'
    filepath = bronze_attributes_directory + partition_name
    df = spark.read.parquet(filepath)
    print('loaded from:', filepath, 'row count:', df.count())

    df = df.drop("Name", "SSN")

    df = df.withColumn("Customer_ID", col("Customer_ID").cast(StringType()))
    df = df.withColumn("snapshot_date", col("snapshot_date").cast(DateType()))

    # FIX: strip trailing underscores/whitespace from Age BEFORE casting to int.
    # Source data has dirty values like '41_' which a direct .cast(IntegerType())
    # cannot parse and throws a hard error under Spark's ANSI mode. We clean the
    # string first, then cast.
    df = df.withColumn(
        "Age",
        regexp_replace(trim(col("Age").cast(StringType())), "[^0-9-]", "").cast(IntegerType())
    )
    df = df.withColumn("Age", F.when((col("Age") >= 18) & (col("Age") <= 100), col("Age")).otherwise(None))
    df = df.fillna({"Age": 35})

    df = df.withColumn("Occupation", col("Occupation").cast(StringType()))
    df = df.fillna({"Occupation": "Unknown"})

    partition_name = "silver_attributes_" + snapshot_date_str.replace('-', '_') + '.parquet'
    filepath = silver_attributes_directory + partition_name
    df.write.mode("overwrite").parquet(filepath)
    print('saved to:', filepath)

    return df


# ─────────────────────────────────────────────
# Financials silver
# ─────────────────────────────────────────────
def process_silver_financials(snapshot_date_str, bronze_financials_directory, silver_financials_directory, spark):
    """
    Clean financials bronze data:
    - Strip trailing underscores from ALL numeric-but-dirty string columns
      (Annual_Income, Outstanding_Debt, and others) before casting
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

    # Helper: strip any trailing/leading non-numeric junk (e.g. '_') from a
    # numeric-looking string column before casting. This handles dirty values
    # like '52312.68_' or '642.42_' that would otherwise throw a hard cast
    # error under Spark's ANSI mode.
    def clean_numeric_string(c):
        cleaned = regexp_replace(trim(col(c).cast(StringType())), "[^0-9.\\-]", "")
        return F.when(F.length(cleaned) > 0, cleaned)

    # Annual_Income
    df = df.withColumn("Annual_Income", clean_numeric_string("Annual_Income").cast(FloatType()))

    # Monthly_Inhand_Salary
    df = df.withColumn("Monthly_Inhand_Salary", clean_numeric_string("Monthly_Inhand_Salary").cast(FloatType()))

    # Outstanding_Debt — same dirty-underscore issue as Annual_Income
    df = df.withColumn("Outstanding_Debt", clean_numeric_string("Outstanding_Debt").cast(FloatType()))

    # Credit_Mix: treat '_' as unknown
    df = df.withColumn("Credit_Mix",
        F.when(trim(col("Credit_Mix").cast(StringType())) == "_", "Unknown")
         .otherwise(col("Credit_Mix").cast(StringType())))
    df = df.fillna({"Credit_Mix": "Unknown"})

    # Type_of_Loan: fill nulls
    df = df.withColumn("Type_of_Loan", col("Type_of_Loan").cast(StringType()))
    df = df.fillna({"Type_of_Loan": "Unknown"})

    # Parse Credit_History_Age: '10 Years and 9 Months' -> integer months (total)
    df = df.withColumn("credit_history_age_str", col("Credit_History_Age").cast(StringType()))
    df = df.withColumn("cha_years",
        F.regexp_extract(col("credit_history_age_str"), r"(\d+)\s+Year", 1).cast(IntegerType()))
    df = df.withColumn("cha_months",
        F.regexp_extract(col("credit_history_age_str"), r"(\d+)\s+Month", 1).cast(IntegerType()))
    df = df.withColumn("Credit_History_Months",
        (F.coalesce(col("cha_years"), F.lit(0)) * 12 + F.coalesce(col("cha_months"), F.lit(0))).cast(IntegerType()))
    df = df.drop("Credit_History_Age", "credit_history_age_str", "cha_years", "cha_months")

    # Cast remaining numeric columns — also clean dirty strings defensively
    # for every numeric column, since the same trailing-underscore pattern
    # could appear in any of them.
    numeric_cols = {
        "Num_Bank_Accounts": IntegerType(),
        "Num_Credit_Card": IntegerType(),
        "Interest_Rate": FloatType(),
        "Num_of_Loan": IntegerType(),
        "Delay_from_due_date": IntegerType(),
        "Num_of_Delayed_Payment": IntegerType(),
        "Changed_Credit_Limit": FloatType(),
        "Num_Credit_Inquiries": FloatType(),
        "Credit_Utilization_Ratio": FloatType(),
        "Total_EMI_per_month": FloatType(),
        "Amount_invested_monthly": FloatType(),
        "Monthly_Balance": FloatType(),
    }
    for c, t in numeric_cols.items():
        df = df.withColumn(c, clean_numeric_string(c).cast(t))

    # Categorical columns that stay as strings
    df = df.withColumn("Payment_of_Min_Amount", col("Payment_of_Min_Amount").cast(StringType()))
    df = df.withColumn("Payment_Behaviour", col("Payment_Behaviour").cast(StringType()))

    # Fill remaining numeric nulls with 0
    numeric_fill_cols = list(numeric_cols.keys()) + ["Annual_Income", "Monthly_Inhand_Salary", "Outstanding_Debt"]
    df = df.fillna(0.0, subset=numeric_fill_cols)

    partition_name = "silver_financials_" + snapshot_date_str.replace('-', '_') + '.parquet'
    filepath = silver_financials_directory + partition_name
    df.write.mode("overwrite").parquet(filepath)
    print('saved to:', filepath)

    return df
