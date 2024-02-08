# coding: utf-8
import click
import mlflow
import numpy as np
import shutil
import pandas as pd
import tensorflow as tf
import os
import tensorflow.keras as keras
from mlflow.data.pandas_dataset import PandasDataset
from itertools import product
from dotenv import load_dotenv


def build_lstm_2_layer_model(window_length=50, future_length=1, n_input_features=7,
                             n_output_features=3, units_lstm_layer=30, dropout_rate=0.2):
    """Builds 2 Layer LSTM-based TF Model in functional API.
    Args:
      window_length: Input Data as Numpy Array, Shape (rows, n_features)
      future_length: Number of time steps that will be predicted in the future.
      n_input_features: Number of features that will be used as Input.
      n_output_features: Number of features that will be predicted.
      units_lstm_layer: Number of Neurons for the LSTM Layers.
      dropout_rate: Dropout Rate for the last Fully Connected Dense Layer.
    Returns:
      keras.models.Model
    """
    inputs = keras.layers.Input(shape=[window_length, n_input_features], dtype=np.float32)

    # Layer1
    lstm1_output, lstm1_state_h, lstm1_state_c = keras.layers.LSTM(units=units_lstm_layer, return_state=True,
                                                                   return_sequences=True)(inputs)
    lstm1_state = [lstm1_state_h, lstm1_state_c]

    # Layer 2
    lstm2_output, lstm2_state_h, lstm2_state_c = keras.layers.LSTM(units=units_lstm_layer, return_state=True,
                                                                   return_sequences=True)(lstm1_output,
                                                                                          initial_state=lstm1_state)

    reshaped = tf.reshape(lstm2_output,
                          [-1, window_length * units_lstm_layer])
    # Dropout
    dropout = tf.keras.layers.Dropout(dropout_rate)(reshaped)

    fc_layer = keras.layers.Dense(n_output_features * future_length, kernel_initializer='he_normal', dtype=tf.float32)(
        dropout)

    output = tf.reshape(fc_layer,
                        [-1, future_length, n_output_features])

    model = keras.models.Model(inputs=[inputs],
                               outputs=[output])
    return model


# Applying Sliding Window
# I will use the TF Data API (https://www.tensorflow.org/guide/data)
# for applying sliding windows at the runtime of training to save memory.
# The function will return a zipped tf.data.Dataset with the following Shapes:
# - x: (batches, window_length, n_features)
# - y: (batches, future_length, n_output_features)
def apply_sliding_window_tf_data_api(train_data_x, batch_size=64, window_length=50, future_length=1,
                                     n_output_features=1, validate=False, shift=1):
    """Applies sliding window on the fly by using the TF Data API.
    Args:
      train_data_x: Input Data as Numpy Array, Shape (rows, n_features)
      batch_size: Batch Size.
      window_length: Window Length or Window Size.
      future_length: Number of time steps that will be predicted in the future.
      n_output_features: Number of features that will be predicted.
      validate: True if input data is a validation set and does not need to be shuffled
      shift: Shifts the Sliding Window by this Parameter.
    Returns:
      tf.data.Dataset
    """

    def make_window_dataset(ds, window_size=5, shift=1, stride=1):
        windows = ds.window(window_size, shift=shift, stride=stride)

        def sub_to_batch(sub):
            return sub.batch(window_size, drop_remainder=True)

        windows = windows.flat_map(sub_to_batch)
        return windows

    X = tf.data.Dataset.from_tensor_slices(train_data_x.astype(np.float32))
    y = tf.data.Dataset.from_tensor_slices(train_data_x[window_length:, :n_output_features].astype(np.float32))
    ds_x = make_window_dataset(X, window_size=window_length, shift=shift, stride=1)
    ds_y = make_window_dataset(y, window_size=future_length, shift=shift, stride=1)

    if not validate:
        train_tf_data = tf.data.Dataset.zip((ds_x, ds_y)).cache().shuffle(buffer_size=200000,
                                                                          reshuffle_each_iteration=True).batch(
            batch_size).prefetch(5)
        return train_tf_data
    else:
        return tf.data.Dataset.zip((ds_x, ds_y)).batch(batch_size).prefetch(1)


# Custom TF Callback to log Metrics by MLflow
class MlflowLogging(tf.keras.callbacks.Callback):
    def __init__(self, **kwargs):
        super().__init__()  # handles base args (e.g., dtype)

    def on_epoch_end(self, epoch, logs=None):
        keys = list(logs.keys())
        for key in keys:
            mlflow.log_metric(str(key), logs.get(key), step=epoch)


@click.command()
@click.option("--window_length", default=50, type=int)
@click.option("--future_length", default=5, type=int)
@click.option("--n_output_features", default=1, type=int)
@click.option("--batch_size", default=64, type=int)
@click.option("--learning_rate", default=0.001, type=float)
def main(window_length, future_length, n_output_features, batch_size, learning_rate):
    # try to load mlflow tracking and artifact uri from environment variables
    tracking_uri = os.environ.get('MLFLOW_TRACKING_URI')
    artifact_uri = os.environ.get('MLFLOW_ARTIFACT_URI')
    print(tracking_uri, artifact_uri)

    # if env variables are not set, load .env file
    if (tracking_uri is None) | (artifact_uri is None):
        load_dotenv()
        print(os.path.expandvars('${MLFLOW_TRACKING_URI}'))
        print(os.path.expandvars('${MLFLOW_ARTIFACT_URI}'))
        tracking_uri = os.path.expandvars('${MLFLOW_TRACKING_URI}'.strip('"\''))
        artifact_uri = os.path.expandvars('${MLFLOW_ARTIFACT_URI}'.strip('"\''))

    mlflow.tracking.set_registry_uri(tracking_uri)
    mlflow.tracking.set_tracking_uri(artifact_uri)

    # The data includes 'nan' and '?' as a string, both will be imported as numpy nan
    # Note that I will only use the first 2000 rows for the example
    source_path = './household_power_consumption.txt'
    df = pd.read_csv(source_path, sep=';',
                     parse_dates={'dt': ['Date', 'Time']}, infer_datetime_format=True,
                     low_memory=False, na_values=['nan', '?'], index_col='dt')

    # filling nan with mean in any columns
    for j in range(0, 7):
        df.iloc[:, j] = df.iloc[:, j].fillna(df.iloc[:, j].mean())

    # Standardization
    mean = df.mean(axis=0)
    std = df.std(axis=0)
    standardized = (df - mean) / std

    # Grid Search Hyperparameter
    # Dictionary with different hyperparameters to train on.
    # MLflow will track those in a database.
    grid_search_dic = {'hidden_layer_size': [20, 40],
                       'batch_size': [batch_size],
                       'future_length': [future_length],
                       'window_length': [window_length],
                       'dropout_fc': [0.0, 0.2],
                       'n_output_features': [n_output_features]}

    # Cartesian product
    grid_search_param = [dict(zip(grid_search_dic, v)) for v in product(*grid_search_dic.values())]

    # Training

    # enable gpu growth if gpu is available
    gpu_devices = tf.config.experimental.list_physical_devices('GPU')
    for device in gpu_devices: tf.config.experimental.set_memory_growth(device, True)

    dataset: PandasDataset = mlflow.data.from_pandas(standardized, source=source_path, targets="Global_active_power")

    with mlflow.start_run() as parent_run:
        mlflow.log_input(dataset, context="training", tags={"version": "1"})
        
        for params in grid_search_param:
            batch_size = params['batch_size']
            window_length = params['window_length']
            future_length = params['future_length']
            dropout_fc = params['dropout_fc']
            hidden_layer_size = params['hidden_layer_size']
            n_output_features = params['n_output_features']

            with mlflow.start_run(nested=True) as child_run:
                # log parameter
                mlflow.log_param('batch_size', batch_size)
                mlflow.log_param('window_length', window_length)
                mlflow.log_param('hidden_layer_size', hidden_layer_size)
                mlflow.log_param('dropout_fc_layer', dropout_fc)
                mlflow.log_param('future_length', future_length)
                mlflow.log_param('n_output_features', n_output_features)

                model = build_lstm_2_layer_model(window_length=window_length,
                                                 future_length=future_length,
                                                 n_output_features=n_output_features,
                                                 units_lstm_layer=hidden_layer_size,
                                                 dropout_rate=dropout_fc)

                data_sliding_window = apply_sliding_window_tf_data_api(standardized.values,
                                                                       batch_size=batch_size,
                                                                       window_length=window_length,
                                                                       future_length=future_length,
                                                                       n_output_features=n_output_features)

                model.compile(loss='mse', optimizer=keras.optimizers.Nadam(learning_rate=learning_rate),
                              metrics=['mse', 'mae'])

                model.fit(data_sliding_window, shuffle=True, initial_epoch=0, epochs=10,
                          callbacks=[MlflowLogging()])

                mlflow.tensorflow.log_model(model=model,
                                            artifact_path='saved_model',
                                            registered_model_name='Electric Power Consumption Forecasting')



if __name__ == "__main__":
    main()
