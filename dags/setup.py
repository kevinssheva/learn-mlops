from airflow.decorators import dag, task, task_group
from pendulum import datetime
from airflow.operators.empty import EmptyOperator
from airflow.providers.amazon.aws.operators.s3 import S3CreateBucketOperator
from mlflow_provider.hooks.client import MLflowClientHook

BASE_DIR = "include"
FILE_PATH = "churn.csv"
PROCESSED_DATA_PATH = "processed_churn.csv"

## AWS S3 parameters
AWS_CONN_ID = "aws_default"
DATA_BUCKET_NAME = "data"
MLFLOW_ARTIFACT_BUCKET = "mlflowdatapossums"

## MLFlow parameters
MLFLOW_CONN_ID = "mlflow_default"
EXPERIMENT_NAME = "Possum_tails"
MAX_RESULTS_MLFLOW_LIST_EXPERIMENTS = 1000

## Data parameters
TARGET_COLUMN = "taill"  # tail length in cm
CATEGORICAL_COLUMNS = ["site", "Pop", "sex"]
NUMERIC_COLUMNS = ["age", "hdlngth", "skullw", "totlngth", "footlgth"]

XCOM_BUCKET = "localxcom"


@dag(
    schedule=None,
    start_date=datetime(2023, 1, 1),
    catchup=False,
)
def experiment_and_bucket_setup():
    start = EmptyOperator(task_id="start")
    end = EmptyOperator(task_id="end")

    create_buckets_if_not_exists = S3CreateBucketOperator.partial(
        task_id="create_buckets_if_not_exists",
        aws_conn_id=AWS_CONN_ID,
    ).expand(bucket_name=[DATA_BUCKET_NAME, MLFLOW_ARTIFACT_BUCKET, XCOM_BUCKET])

    @task_group
    def prepare_mlflow_experiment():
        @task
        def list_existing_experiments(max_results=1000):
            mlflow_hook = MLflowClientHook(mlflow_conn_id=MLFLOW_CONN_ID)
            return mlflow_hook.run(
                endpoint="api/2.0/mlflow/experiments/search",
                request_params={"max_results": max_results},
            ).json()

        @task.branch
        def check_if_experiment_exists(
            experiment_name, existing_experiments_information
        ):
            if existing_experiments_information:
                existing_experiment_names = [
                    experiment["name"]
                    for experiment in existing_experiments_information["experiments"]
                ]
                return (
                    "prepare_mlflow_experiment.experiment_exists"
                    if experiment_name in existing_experiment_names
                    else "prepare_mlflow_experiment.create_experiment"
                )
            return "prepare_mlflow_experiment.create_experiment"

        @task
        def create_experiment(experiment_name, artifact_bucket):
            mlflow_hook = MLflowClientHook(mlflow_conn_id=MLFLOW_CONN_ID)
            return mlflow_hook.run(
                endpoint="api/2.0/mlflow/experiments/create",
                request_params={
                    "name": experiment_name,
                    "artifact_location": f"s3://{artifact_bucket}/",
                },
            ).json()

        experiment_already_exists = EmptyOperator(task_id="experiment_exists")

        (
            check_if_experiment_exists(
                experiment_name=EXPERIMENT_NAME,
                existing_experiments_information=list_existing_experiments(),
            )
            >> [
                experiment_already_exists,
                create_experiment(EXPERIMENT_NAME, MLFLOW_ARTIFACT_BUCKET),
            ]
        )

    start >> create_buckets_if_not_exists >> prepare_mlflow_experiment() >> end


experiment_and_bucket_setup()
