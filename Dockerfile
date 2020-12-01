FROM python:3.7-slim-buster

WORKDIR /

# MLflow will automatically try to use LibYAML bindings if they are already installed.
RUN apt-get update -y
RUN apt-get install libyaml-cpp-dev libyaml-dev -y

# Install python packages
RUN pip install mlflow pymysql

RUN pip --no-cache-dir install --force-reinstall -I pyyaml

ENTRYPOINT ["mlflow", "server"]