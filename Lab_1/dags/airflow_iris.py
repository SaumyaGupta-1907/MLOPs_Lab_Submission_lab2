# Import necessary libraries and modules
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from datetime import datetime, timedelta

from src.lab_iris import load_iris_data, preprocess_data, train_mlp, evaluate_mlp

# Enable pickle support for XCom
from airflow import configuration as conf
conf.set('core', 'enable_xcom_pickling', 'True')

# Default DAG arguments
default_args = {
    'owner': 'your_name',
    'start_date': datetime(2025, 1, 15),
    'retries': 0,
    'retry_delay': timedelta(minutes=5),
}

# Define the DAG
dag = DAG(
    'iris_mlp_training',
    default_args=default_args,
    schedule_interval=None,
    catchup=False,
    description='Train and evaluate MLP on Iris dataset with notifications',
)

t1 = PythonOperator(
    task_id="load_data",
    python_callable=load_iris_data,
    dag=dag,
)

t2 = PythonOperator(
    task_id="preprocess_data",
    python_callable=preprocess_data,
    op_args=[t1.output],
    dag=dag,
)
# Task 3: train MLP, gets output from t2 

t3 = PythonOperator(task_id="train_mlp", python_callable=train_mlp, op_args=[t2.output, 32], dag=dag, )

t4 = PythonOperator(
    task_id="evaluate_mlp",
    python_callable=evaluate_mlp,
    op_args=[t3.output],
    dag=dag,
)

t1 >> t2 >> t3 >> t4

# CLI entry point
if __name__ == "__main__":
    dag.cli()

