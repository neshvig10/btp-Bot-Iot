import argparse
import warnings
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split as sklearn_train_test_split
import flwr as fl
import pickle
from flwr.common import Code, EvaluateIns, EvaluateRes, FitIns, FitRes, GetParametersIns, GetParametersRes, Parameters, Status

warnings.filterwarnings("ignore", category=UserWarning)

# Define arguments parser for the client/partition ID.
parser = argparse.ArgumentParser()
parser.add_argument(
    "--partition-id",
    default=0,
    type=int,
    help="Partition ID used for the current client.",
)
args = parser.parse_args()

# Define data partitioning related functions
def custom_train_test_split(data: pd.DataFrame, test_fraction: float, seed: int):
    """Split the data into train and validation set given split rate."""
    train_data, valid_data = sklearn_train_test_split(data, test_size=test_fraction, random_state=seed)
    num_train = len(train_data)
    num_val = len(valid_data)
    return train_data, valid_data, num_train, num_val

# Load CSV dataset
data = pd.read_csv("./UNSW_2018_IoT_Botnet_Dataset_74.csv")

# Train/test splittings
train_data, valid_data, num_train, num_val = custom_train_test_split(data, test_fraction=0.2, seed=42)

# Split into features and labels
X_train, y_train = train_data.drop(columns=['T']).values, train_data['T'].values
X_valid, y_valid = valid_data.drop(columns=['T']).values, valid_data['T'].values

# Define a Keras Sequential model
def create_keras_model(input_shape):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(128, activation='relu', input_shape=input_shape),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['RootMeanSquaredError'])
    return model

# Initialize Keras model
model = create_keras_model((X_train.shape[1],))

# Define Flower client
class KerasClient(fl.client.Client):
    def __init__(self):
        self.model = model

    def get_parameters(self, ins: GetParametersIns) -> GetParametersRes:
        # Serialize the model weights (list) into bytes
        weights = self.model.get_weights()
        serialized_weights = pickle.dumps(weights)  # Convert list to bytes
        
        return GetParametersRes(
            status=Status(
                code=Code.OK,
                message="OK",
            ),
            parameters=Parameters(
                tensors=[serialized_weights],  # Return serialized weights as a tensor
                tensor_type="",
            ),
        )

    def fit(self, ins: FitIns) -> FitRes:
        # Deserialize the model weights from bytes back to a list
        serialized_weights = ins.parameters.tensors[0]
        weights = pickle.loads(serialized_weights)  # Convert bytes back to list
        
        # Set the model weights
        self.model.set_weights(weights)

        # Perform local training
        self.model.fit(X_train, y_train, epochs=5, batch_size=32, validation_data=(X_valid, y_valid))

        # After training, serialize the updated weights to send back to the server
        updated_weights = self.model.get_weights()
        serialized_updated_weights = pickle.dumps(updated_weights)

        return FitRes(
            status=Status(
                code=Code.OK,
                message="OK",
            ),
            parameters=Parameters(
                tensors=[serialized_updated_weights],  # Send updated weights as bytes
                tensor_type="",
            ),
            num_examples=len(X_train),
            metrics={},
        )

    def evaluate(self, ins: EvaluateIns) -> EvaluateRes:
        loss, rmse = self.model.evaluate(X_valid, y_valid, verbose=0)
        
        # Calculate R-squared
        y_pred = self.model.predict(X_valid)
        ss_res = np.sum((y_valid - y_pred.flatten()) ** 2)
        ss_tot = np.sum((y_valid - np.mean(y_valid)) ** 2)
        r_squared = 1 - (ss_res / ss_tot)

        return EvaluateRes(
            status=Status(
                code=Code.OK,
                message="OK",
            ),
            loss=loss,
            num_examples=len(X_valid),
            metrics={"RMSE": rmse, "R-squared": r_squared},
        )

# Start Flower client
fl.client.start_client(server_address="127.0.0.1:8080", client=KerasClient())
