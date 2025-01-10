from airflow import Dataset
from airflow.decorators import dag, task, task_group
from pendulum import datetime
from astro import sql as aql
from astro.files import File
from astro.dataframes.pandas import DataFrame
from airflow.operators.empty import EmptyOperator
from airflow.providers.amazon.aws.operators.s3 import S3CreateBucketOperator
from mlflow_provider.hooks.client import MLflowClientHook
import os
import pandas as pd

# Constants used in the DAG
FILE_PATH = "churn.csv"
PROCESSED_TRAIN_DATA_PATH = "train_data.csv"

# AWS S3 parameters
AWS_CONN_ID = "aws_default"
DATA_BUCKET_NAME = "data"
MLFLOW_ARTIFACT_BUCKET = "mlflowdatachurn"
XCOM_BUCKET = "localxcom"

# MLFlow parameters
MLFLOW_CONN_ID = "mlflow_default"
EXPERIMENT_NAME = "customer_churn"
MAX_RESULTS_MLFLOW_LIST_EXPERIMENTS = 100

# Data parameters
NUMERIC_COLUMNS = ["tenure", "MonthlyCharges", "TotalCharges"]


@dag(
    schedule=None,
    start_date=datetime(2023, 1, 1),
    catchup=False,
)
def data_preparation():
    start = EmptyOperator(task_id="start")
    end = EmptyOperator(
        task_id="end",
        outlets=[Dataset("s3://" + DATA_BUCKET_NAME + "_" + PROCESSED_TRAIN_DATA_PATH)],
    )

    create_buckets_if_not_exists = S3CreateBucketOperator.partial(
        task_id="create_buckets_if_not_exists",
        aws_conn_id=AWS_CONN_ID,
    ).expand(bucket_name=[DATA_BUCKET_NAME, MLFLOW_ARTIFACT_BUCKET, XCOM_BUCKET])

    @task_group
    def prepare_mlflow_experiment():
        @task
        def list_existing_experiments(max_results=1000):
            "Get information about existing MLFlow experiments."

            mlflow_hook = MLflowClientHook(mlflow_conn_id=MLFLOW_CONN_ID)
            existing_experiments_information = mlflow_hook.run(
                endpoint="api/2.0/mlflow/experiments/search",
                request_params={"max_results": max_results},
            ).json()

            return existing_experiments_information

        @task.branch
        def check_if_experiment_exists(
            experiment_name, existing_experiments_information
        ):
            "Check if the specified experiment already exists."

            if existing_experiments_information:
                existing_experiment_names = [
                    experiment["name"]
                    for experiment in existing_experiments_information["experiments"]
                ]
                if experiment_name in existing_experiment_names:
                    return "prepare_mlflow_experiment.experiment_exists"
                else:
                    return "prepare_mlflow_experiment.create_experiment"
            else:
                return "prepare_mlflow_experiment.create_experiment"

        @task
        def create_experiment(experiment_name, artifact_bucket):
            """Create a new MLFlow experiment with a specified name.
            Save artifacts to the specified S3 bucket."""

            mlflow_hook = MLflowClientHook(mlflow_conn_id=MLFLOW_CONN_ID)
            new_experiment_information = mlflow_hook.run(
                endpoint="api/2.0/mlflow/experiments/create",
                request_params={
                    "name": experiment_name,
                    "artifact_location": f"s3://{artifact_bucket}/",
                },
            ).json()

            return new_experiment_information

        experiment_already_exists = EmptyOperator(task_id="experiment_exists")

        @task(
            trigger_rule="none_failed",
        )
        def get_current_experiment_id(experiment_name, max_results=1000):
            "Get the ID of the specified MLFlow experiment."

            mlflow_hook = MLflowClientHook(mlflow_conn_id=MLFLOW_CONN_ID)
            experiments_information = mlflow_hook.run(
                endpoint="api/2.0/mlflow/experiments/search",
                request_params={"max_results": max_results},
            ).json()

            for experiment in experiments_information["experiments"]:
                if experiment["name"] == experiment_name:
                    return experiment["experiment_id"]

            raise ValueError(f"{experiment_name} not found in MLFlow experiments.")

        experiment_id = get_current_experiment_id(
            experiment_name=EXPERIMENT_NAME,
            max_results=MAX_RESULTS_MLFLOW_LIST_EXPERIMENTS,
        )

        (
            check_if_experiment_exists(
                experiment_name=EXPERIMENT_NAME,
                existing_experiments_information=list_existing_experiments(
                    max_results=MAX_RESULTS_MLFLOW_LIST_EXPERIMENTS
                ),
            )
            >> [
                experiment_already_exists,
                create_experiment(
                    experiment_name=EXPERIMENT_NAME,
                    artifact_bucket=MLFLOW_ARTIFACT_BUCKET,
                ),
            ]
            >> experiment_id
        )

    @aql.dataframe()
    def data_manipulation(data_file: str) -> DataFrame:
        try:
            df = pd.read_csv(f"include/{data_file}")
        except FileNotFoundError:
            raise ValueError(f"File {data_file} not found in the specified directory.")

        df = df.drop(["customerID"], axis=1)
        df["TotalCharges"] = pd.to_numeric(df.TotalCharges, errors="coerce")
        df.drop(labels=df[df["tenure"] == 0].index, axis=0, inplace=True)
        df["TotalCharges"].fillna(df["TotalCharges"].mean(), inplace=True)
        df["SeniorCitizen"] = df["SeniorCitizen"].map({0: "No", 1: "Yes"})

        return df

    @aql.dataframe()
    def data_preprocessing(
        preprocessed_df: DataFrame,
        experiment_id: str,
    ):
        """Scale features and log preprocessed data to MLFlow."""
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler
        from sklearn.preprocessing import LabelEncoder
        import mlflow
        import numpy as np

        mlflow.sklearn.autolog()

        categorical_columns = preprocessed_df.select_dtypes(include=["object"]).columns
        preprocessed_df[categorical_columns] = preprocessed_df[
            categorical_columns
        ].apply(LabelEncoder().fit_transform)

        scaler = StandardScaler()

        # Start an MLFlow run
        with mlflow.start_run(experiment_id=experiment_id, run_name="Scaler"):
            numeric_data = preprocessed_df[NUMERIC_COLUMNS].values
            preprocessed_df[NUMERIC_COLUMNS] = scaler.fit_transform(numeric_data)
            mlflow.sklearn.log_model(scaler, artifact_path="scaler")
            mlflow.log_metrics(
                {"mean_feature_value": np.mean(preprocessed_df[NUMERIC_COLUMNS].values)}
            )
        return preprocessed_df

    # Task execution flow
    preprocessed_df = data_manipulation(data_file=FILE_PATH)

    processed_df = data_preprocessing(
        preprocessed_df=preprocessed_df,
        experiment_id="{{ ti.xcom_pull(task_ids='prepare_mlflow_experiment.get_current_experiment_id') }}",
    )

    save_data_to_s3 = aql.export_file(
        task_id="save_data_to_s3",
        input_data=processed_df,
        output_file=File(
            path=os.path.join("s3://", DATA_BUCKET_NAME, PROCESSED_TRAIN_DATA_PATH),
            conn_id=AWS_CONN_ID,
        ),
        if_exists="replace",
    )

    # Task dependencies
    (
        start
        >> create_buckets_if_not_exists
        >> prepare_mlflow_experiment()
        >> preprocessed_df
        >> processed_df
        >> save_data_to_s3
        >> end
    )


data_preparation()
