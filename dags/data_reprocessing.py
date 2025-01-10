from airflow import Dataset
from airflow.decorators import dag, task
from pendulum import datetime
from astro import sql as aql
from astro.files import File
from astro.dataframes.pandas import DataFrame
from airflow.operators.empty import EmptyOperator
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Constants used in the DAG
COMBINED_DATA_PATH = "combined_data.csv"
PROCESSED_COMBINED_DATA_PATH = "processed_combined_data.csv"

# AWS S3 parameters
AWS_CONN_ID = "aws_default"
DATA_BUCKET_NAME = "data"

# Data parameters
NUMERIC_COLUMNS = ["tenure", "MonthlyCharges", "TotalCharges"]


@dag(
    schedule=[Dataset("s3://" + DATA_BUCKET_NAME + "_" + COMBINED_DATA_PATH)],
    start_date=datetime(2023, 1, 1),
    catchup=False,
)
def data_reprocessing():
    start = EmptyOperator(task_id="start")
    end = EmptyOperator(
        task_id="end",
        outlets=[
            Dataset("s3://" + DATA_BUCKET_NAME + "_" + PROCESSED_COMBINED_DATA_PATH)
        ],
    )

    @aql.dataframe()
    def fetch_data(file_path: str) -> DataFrame:
        """Fetch data from S3."""
        return pd.read_csv(f"s3://{DATA_BUCKET_NAME}/{file_path}")

    @aql.dataframe()
    def data_manipulation(df: DataFrame) -> DataFrame:
        """Clean and preprocess raw data."""
        df = df.drop(["customerID"], axis=1)
        df["TotalCharges"] = pd.to_numeric(df.TotalCharges, errors="coerce")
        df.drop(labels=df[df["tenure"] == 0].index, axis=0, inplace=True)
        df["TotalCharges"].fillna(df["TotalCharges"].mean(), inplace=True)
        df["SeniorCitizen"] = df["SeniorCitizen"].map({0: "No", 1: "Yes"})
        return df

    @aql.dataframe()
    def data_preprocessing(preprocessed_df: DataFrame) -> DataFrame:
        """Scale numeric features and encode categorical features."""
        categorical_columns = preprocessed_df.select_dtypes(include=["object"]).columns
        preprocessed_df[categorical_columns] = preprocessed_df[
            categorical_columns
        ].apply(LabelEncoder().fit_transform)

        scaler = StandardScaler()
        preprocessed_df[NUMERIC_COLUMNS] = scaler.fit_transform(
            preprocessed_df[NUMERIC_COLUMNS]
        )
        return preprocessed_df

    # Task execution flow
    raw_df = fetch_data(file_path=COMBINED_DATA_PATH)

    preprocessed_df = data_manipulation(df=raw_df)

    processed_df = data_preprocessing(preprocessed_df=preprocessed_df)

    save_data_to_s3 = aql.export_file(
        task_id="save_data_to_s3",
        input_data=processed_df,
        output_file=File(
            path=f"s3://{DATA_BUCKET_NAME}/{PROCESSED_COMBINED_DATA_PATH}",
            conn_id=AWS_CONN_ID,
        ),
        if_exists="replace",
    )

    # Task dependencies
    start >> raw_df >> preprocessed_df >> processed_df >> save_data_to_s3 >> end


data_reprocessing()
