import mlflow
import mlflow.sklearn
from ingest_data import main as ingest_data_main
from score import main as score_main

# from HousePricePrediction.ingest_data import main as ingest_data_main
# from HousePricePrediction.score import main as score_main
# from HousePricePrediction.train import main as train_main
from train import main as train_main

remote_server_uri = "http://127.0.0.1:5000/"  # set to your server URI
mlflow.set_tracking_uri(remote_server_uri)  # or set the MLFLOW_TRACKING_URI in the env

print(mlflow.tracking.get_tracking_uri())
exp_name = "House_Price_Prediction"
mlflow.set_experiment(exp_name)

# Create nested runs
# experiment_id = mlflow.create_experiment("experiment1")
with mlflow.start_run(
    run_name="PARENT_RUN",
    tags={"version": "v1", "priority": "P1"},
    description="parent",
) as parent_run:
    mlflow.log_param("parent", "yes")
    with mlflow.start_run(
        run_name="CHILD_RUN_INGEST",
        description="Ingest Data",
        nested=True,
    ) as child_run:
        mlflow.log_param("child1", "yes")
        ingest = ingest_data_main()
    with mlflow.start_run(
        run_name="CHILD_TRAIN_MODELS",
        description="Training ML models",
        nested=True,
    ) as child_run:
        mlflow.log_param("child2", "yes")
        output1 = train_main()
    with mlflow.start_run(
        run_name="CHILD_COMPUTE_SCORE",
        description="Compute scores for the models",
        nested=True,
    ) as child_run:
        mlflow.log_param("child3", "yes")
        output2 = score_main()


print("parent run:")

print("run_id: {}".format(parent_run.info.run_id))
print("description: {}".format(parent_run.data.tags.get("mlflow.note.content")))
print("version tag value: {}".format(parent_run.data.tags.get("version")))
print("priority tag value: {}".format(parent_run.data.tags.get("priority")))
print("--")
