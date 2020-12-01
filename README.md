# MLflow example to track Parameter and Metrics by using MLproject Functionality
This MLflow example uses a simple LSTM-based time series forecasting model in TensorFlow 2 to demonstrate the Tracking and MLproject Functionality.<br>
The Dataset can be found here: https://archive.ics.uci.edu/ml/datasets/individual+household+electric+power+consumption <br>
Related Blog Post: [https://www.novatec-gmbh.de/blog/mlflow-tracking-von-parametern-und-metriken/](https://www.novatec-gmbh.de/blog/mlflow-tracking-von-parametern-und-metriken/)
## Requirements
- Mlflows MLproject will build a conda environment from the conda.yaml file at runtime. Therefore [Anaconda](https://www.anaconda.com/distribution/) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html) must be installed.<br>
- MLflow must be installed: ```pip install mlflow```
- MLflow will look for a git executable to track the git commit for every experiment. To disable this: ```export GIT_PYTHON_REFRESH=quiet```
## Quick Start
#### MLflow Server with Docker
If docker and docker-compose is installed, use docker-compose.yaml to start a MLflow server and MySQL backend:<br>
```docker-compose up -d``` <br>
Set environment variables for MLflow:<br>
```export MLFLOW_TRACKING_URI=mysql+pymysql://mlflow:mlflow@localhost:3306/mlflow``` <br>
```export MLFLOW_ARTIFACT_URI=http://localhost:5000```<br>
Start the MLflow Experiments:<br>
```mlflow run .```

#### MLflow Server with SQLite
To use a SQLite backend and to start a local MLflow server, change the directory to the repository and use:<br>
```mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns --host 0.0.0.0``` <br>
Set environment variables for MLflow:<br>
```export MLFLOW_TRACKING_URI=sqlite:///mlflow.db```<br>
```export MLFLOW_ARTIFACT_URI=http://localhost:5000```<br>
Start the MLflow Experiments:<br>
```mlflow run .```

## Access the MLflow Dashboard: [http://localhost:5000](http://localhost:5000)

# Jupyter Notebook
There is a Jupyter Notebook available to explore the code in detail. For this purpose create a new conda environment from the conda.yaml, activate the environment and start a jupyter server.