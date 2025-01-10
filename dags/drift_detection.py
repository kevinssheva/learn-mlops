from airflow import Dataset
from airflow.decorators import dag, task
from pendulum import datetime
from astro import sql as aql
from astro.files import File
from astro.dataframes.pandas import DataFrame
from airflow.operators.empty import EmptyOperator

import os
import pandas as pd
import numpy as np

# Constants
BASELINE_DATA_PATH = "churn.csv"
PRODUCTION_DATA_PATH = "churn_production.csv"
COMBINED_DATA_PATH = "churn_combined.csv"
AWS_CONN_ID = "aws_default"
DATA_BUCKET_NAME = "data"
PSI_THRESHOLD = 0.00001
NUMERIC_COLUMNS = ["tenure", "MonthlyCharges", "TotalCharges"]


@dag(
    schedule=None,
    start_date=datetime(2023, 1, 1),
    catchup=False,
)
def drift_detection():
    start = EmptyOperator(task_id="start")
    end = EmptyOperator(
        task_id="end",
        outlets=[Dataset("s3://" + DATA_BUCKET_NAME + "_" + COMBINED_DATA_PATH)],
    )
    skip_saving = EmptyOperator(task_id="skip_saving", trigger_rule="none_failed")

    @aql.dataframe()
    def extract_data(file_path) -> DataFrame:
        df = pd.read_csv(f"include/{file_path}")
        return df

    @task
    def calculate_psi(
        baseline_df: pd.DataFrame, production_df: pd.DataFrame, columns: list
    ) -> dict:
        """Calculate PSI for specified columns."""

        def preprocess_column(df, col):
            """Ensure column is numeric and fill NaN values."""
            df[col] = pd.to_numeric(df[col], errors="coerce")  # Convert to numeric
            df[col] = df[col].fillna(df[col].mean())  # Replace NaN with column mean

        def calculate_column_psi(base_col, prod_col):
            base_hist, bin_edges = np.histogram(base_col, bins=10, density=True)
            prod_hist, _ = np.histogram(prod_col, bins=bin_edges, density=True)
            psi_values = (prod_hist - base_hist) * np.log(
                (prod_hist + 1e-6) / (base_hist + 1e-6)
            )
            return np.sum(psi_values)

        for col in columns:
            preprocess_column(baseline_df, col)
            preprocess_column(production_df, col)

        psi_scores = {
            col: calculate_column_psi(baseline_df[col], production_df[col])
            for col in columns
        }
        return psi_scores

    @task.branch
    def check_drift(psi_scores: dict, threshold: float) -> str:
        """Check if data drift occurred and branch tasks based on the result."""
        drift_detected = any(score > threshold for score in psi_scores.values())
        print(f"Drift detected: {drift_detected}")
        return "save_data_to_s3" if drift_detected else "skip_saving"

    @task
    def combine_data(
        baseline_df: pd.DataFrame, production_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Combine baseline and production datasets."""
        combined_df = pd.concat([baseline_df, production_df], ignore_index=True)
        return combined_df

    baseline_data = extract_data(BASELINE_DATA_PATH)
    production_data = extract_data(PRODUCTION_DATA_PATH)

    psi_scores = calculate_psi(baseline_data, production_data, NUMERIC_COLUMNS)

    next_task = check_drift(psi_scores, PSI_THRESHOLD)

    combined_data = combine_data(baseline_data, production_data)

    save_combined_data_to_s3 = aql.export_file(
        task_id="save_data_to_s3",
        input_data=combined_data,
        output_file=File(
            path=os.path.join("s3://", DATA_BUCKET_NAME, COMBINED_DATA_PATH),
            conn_id=AWS_CONN_ID,
        ),
        if_exists="replace",
    )

    # Task dependencies
    start >> [baseline_data, production_data] >> psi_scores >> next_task
    next_task >> save_combined_data_to_s3 >> end
    next_task >> skip_saving >> end


drift_detection()
