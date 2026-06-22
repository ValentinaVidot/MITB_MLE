import os
from datetime import datetime
import pyspark.sql.functions as F
from pyspark.sql.functions import col
from pyspark.sql.types import StringType, IntegerType, FloatType, DateType


# ─────────────────────────────────────────────
# EXISTING FROM LAB 2 — Label store gold
# ─────────────────────────────────────────────
def process_labels_gold_table(snapshot_date_str, silver_loan_daily_directory, gold_label_store_directory, spark, dpd, mob):
    """
    Build label store gold table.
    Label = 1 if customer had DPD >= dpd at MOB == mob, else 0.
    This is the ML target variable (Y).

    IMPORTANT: 'snapshot_date' here is the date this loan/customer row was
    observed at MOB == mob (e.g. 6 months after loan origination) — it is
    NOT the loan application date. We additionally carry 'loan_start_date'
    (renamed to 'feature_snapshot_date') because that is the date features
    must be joined on: the point-in-time of loan APPLICATION, when the bank
    actually had to make its lending decision. Joining on the MOB6 date
    instead would both (a) never match the feature store's per-month
    snapshots, since few loans happen to start exactly 6 months before a
    feature snapshot, and (b) leak 6 months of future information into
    the model if it accidentally did match.
    """
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")

    partition_name = "silver_loan_daily_" + snapshot_date_str.replace('-', '_') + '.parquet'
    filepath = silver_loan_daily_directory + partition_name
    df = spark.read.parquet(filepath)
    print('loaded from:', filepath, 'row count:', df.count())

    df = df.filter(col("mob") == mob)
    df = df.withColumn("label", F.when(col("dpd") >= dpd, 1).otherwise(0).cast(IntegerType()))
    df = df.withColumn("label_def", F.lit(str(dpd) + 'dpd_' + str(mob) + 'mob').cast(StringType()))

    # feature_snapshot_date = loan application date = the date to join features on
    df = df.withColumn("feature_snapshot_date", col("loan_start_date").cast(DateType()))

    df = df.select("loan_id", "Customer_ID", "label", "label_def", "feature_snapshot_date", "snapshot_date")

    partition_name = "gold_label_store_" + snapshot_date_str.replace('-', '_') + '.parquet'
    filepath = gold_label_store_directory + partition_name
    df.write.mode("overwrite").parquet(filepath)
    print('saved to:', filepath)

    return df


# ─────────────────────────────────────────────
# UPGRADED — Feature store gold (Assignment 2)
# ─────────────────────────────────────────────
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

    Assignment 2 upgrades over Assignment 1:
    1. Financial ratio features (debt_to_income, emi_to_salary, repayment_ability,
       loans_per_credit_item, num_fin_pdts) — inspired by classmates' work, these
       are standard credit-risk metrics that capture relationships between raw
       numbers rather than just the raw numbers themselves.
    2. One-hot encoding of key categoricals (Credit_Mix, Payment_Behaviour,
       Occupation) — models can't use text directly; one-hot avoids implying
       false ordinal relationships between categories.
    3. Type_of_Loan split into individual binary flag columns (has_mortgage_loan,
       has_auto_loan, etc.) — preserves signal that would be lost if kept as a
       single comma-separated string.

    Design choices kept from Assignment 1:
    - Left join: attributes (base) -> financials -> clickstream
    - No LMS repayment data joined in (avoids target leakage)
    - snapshot_date represents point-in-time of the feature snapshot
    """
    # ── Load silver tables ──────────────────────────────────────────────
    attr_partition = "silver_attributes_" + snapshot_date_str.replace('-', '_') + '.parquet'
    attr_df = spark.read.parquet(silver_attributes_directory + attr_partition)

    fin_partition = "silver_financials_" + snapshot_date_str.replace('-', '_') + '.parquet'
    fin_df = spark.read.parquet(silver_financials_directory + fin_partition)

    click_partition = "silver_clickstream_" + snapshot_date_str.replace('-', '_') + '.parquet'
    click_df = spark.read.parquet(silver_clickstream_directory + click_partition)

    print(f'{snapshot_date_str} | attributes: {attr_df.count()} | financials: {fin_df.count()} | clickstream: {click_df.count()}')

    # ══════════════════════════════════════════════════════════════════
    # NEW: Financial ratio feature engineering (done before join)
    # ══════════════════════════════════════════════════════════════════
    fin_df = fin_df.withColumn(
        "num_fin_pdts",
        (F.coalesce(col("Num_Bank_Accounts"), F.lit(0)) +
         F.coalesce(col("Num_Credit_Card"), F.lit(0)) +
         F.coalesce(col("Num_of_Loan"), F.lit(0))).cast(IntegerType())
    )

    fin_df = fin_df.withColumn(
        "loans_per_credit_item",
        F.when(
            (col("Num_Bank_Accounts") + col("Num_Credit_Card")) > 0,
            col("Num_of_Loan") / (col("Num_Bank_Accounts") + col("Num_Credit_Card"))
        ).otherwise(0.0).cast(FloatType())
    )

    fin_df = fin_df.withColumn(
        "debt_to_income",
        F.when(
            col("Monthly_Inhand_Salary") > 0,
            col("Outstanding_Debt") / col("Monthly_Inhand_Salary")
        ).otherwise(0.0).cast(FloatType())
    )

    fin_df = fin_df.withColumn(
        "emi_to_salary",
        F.when(
            col("Monthly_Inhand_Salary") > 0,
            col("Total_EMI_per_month") / col("Monthly_Inhand_Salary")
        ).otherwise(0.0).cast(FloatType())
    )

    fin_df = fin_df.withColumn(
        "repayment_ability",
        (col("Monthly_Inhand_Salary") - col("Total_EMI_per_month")).cast(FloatType())
    )

    fin_df = fin_df.withColumn(
        "loan_extent",
        F.when(
            col("Num_of_Loan") > 0,
            col("Delay_from_due_date") / col("Num_of_Loan")
        ).otherwise(0.0).cast(FloatType())
    )

    # ══════════════════════════════════════════════════════════════════
    # NEW: Split Type_of_Loan into binary flag columns
    # ══════════════════════════════════════════════════════════════════
    loan_types = [
        "Auto Loan", "Credit-Builder Loan", "Personal Loan", "Home Equity Loan",
        "Mortgage Loan", "Student Loan", "Debt Consolidation Loan", "Payday Loan"
    ]
    for lt in loan_types:
        col_name = "has_" + lt.lower().replace(" ", "_").replace("-", "_")
        fin_df = fin_df.withColumn(
            col_name,
            F.when(col("Type_of_Loan").contains(lt), 1).otherwise(0).cast(IntegerType())
        )
    fin_df = fin_df.drop("Type_of_Loan")

    # ══════════════════════════════════════════════════════════════════
    # NEW: One-hot encode key categoricals
    # ══════════════════════════════════════════════════════════════════
    # Credit_Mix: Good / Standard / Bad / Unknown
    credit_mix_values = ["Good", "Standard", "Bad", "Unknown"]
    for v in credit_mix_values:
        fin_df = fin_df.withColumn(
            "credit_mix_" + v.lower(),
            F.when(col("Credit_Mix") == v, 1).otherwise(0).cast(IntegerType())
        )
    fin_df = fin_df.drop("Credit_Mix")

    # Payment_Behaviour: keep top categories, one-hot encode
    payment_behaviour_values = [
        "Low_spent_Small_value_payments", "Low_spent_Medium_value_payments",
        "Low_spent_Large_value_payments", "High_spent_Small_value_payments",
        "High_spent_Medium_value_payments", "High_spent_Large_value_payments"
    ]
    for v in payment_behaviour_values:
        col_name = "payment_behaviour_" + v.lower()
        fin_df = fin_df.withColumn(
            col_name,
            F.when(col("Payment_Behaviour") == v, 1).otherwise(0).cast(IntegerType())
        )
    fin_df = fin_df.drop("Payment_Behaviour")

    # Occupation: one-hot encode (from attributes)
    occupation_values = [
        "Lawyer", "Architect", "Engineer", "Scientist", "Mechanic", "Accountant",
        "Developer", "Media_Manager", "Teacher", "Entrepreneur", "Doctor",
        "Journalist", "Manager", "Musician", "Writer", "Unknown"
    ]
    for v in occupation_values:
        col_name = "occupation_" + v.lower()
        attr_df = attr_df.withColumn(
            col_name,
            F.when(col("Occupation") == v, 1).otherwise(0).cast(IntegerType())
        )
    attr_df = attr_df.drop("Occupation")

    # ── Drop snapshot_date from non-base tables to avoid join collision ──
    fin_df = fin_df.drop("snapshot_date")
    click_df = click_df.drop("snapshot_date")

    # ── Join: attributes is the base (left), join financials and clickstream ──
    df = attr_df \
        .join(fin_df, on="Customer_ID", how="left") \
        .join(click_df, on="Customer_ID", how="left")

    # ── Fill nulls introduced by left joins ──
    fe_columns = [f"fe_{i}" for i in range(1, 21)]
    df = df.fillna(0, subset=fe_columns)

    numeric_financial_cols = [
        "Annual_Income", "Monthly_Inhand_Salary", "Num_Bank_Accounts",
        "Num_Credit_Card", "Interest_Rate", "Num_of_Loan",
        "Delay_from_due_date", "Num_of_Delayed_Payment", "Changed_Credit_Limit",
        "Num_Credit_Inquiries", "Outstanding_Debt", "Credit_Utilization_Ratio",
        "Total_EMI_per_month", "Amount_invested_monthly", "Monthly_Balance",
        "Credit_History_Months", "num_fin_pdts", "loans_per_credit_item",
        "debt_to_income", "emi_to_salary", "repayment_ability", "loan_extent"
    ]
    df = df.fillna(0.0, subset=numeric_financial_cols)

    payment_of_min_amount_col = ["Payment_of_Min_Amount"]
    df = df.fillna("Unknown", subset=[c for c in payment_of_min_amount_col if c in df.columns])

    # ── Final schema enforcement ──
    df = df.withColumn("snapshot_date", col("snapshot_date").cast(DateType()))
    df = df.withColumn("Customer_ID", col("Customer_ID").cast(StringType()))

    print(f'{snapshot_date_str} | feature store row count: {df.count()} | columns: {len(df.columns)}')

    # ── Save gold feature store ──
    partition_name = "gold_feature_store_" + snapshot_date_str.replace('-', '_') + '.parquet'
    filepath = gold_feature_store_directory + partition_name
    df.write.mode("overwrite").parquet(filepath)
    print('saved to:', filepath)

    return df
