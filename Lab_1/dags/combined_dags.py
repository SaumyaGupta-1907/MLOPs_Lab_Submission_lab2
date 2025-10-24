from airflow import DAG
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
from datetime import datetime, timedelta

default_args = {
    'owner': 'you',
    'start_date': datetime(2025, 1, 15),
    'retries': 0,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    dag_id='lab1_and_iris_sequential',
    default_args=default_args,
    schedule_interval=None,   # manual trigger
    catchup=False,
)


# Trigger Airflow_Lab1 first
trigger_lab1 = TriggerDagRunOperator(
    task_id='trigger_airflow_lab1',
    trigger_dag_id='Airflow_Lab1',
    wait_for_completion=True,         # wait until it finishes
    dag=dag,
)

# Then trigger iris_mlp_training
trigger_iris = TriggerDagRunOperator(
    task_id='trigger_iris_mlp_training',
    trigger_dag_id='iris_mlp_training',
    wait_for_completion=True,
    dag=dag,
)

trigger_lab1 >> trigger_iris

