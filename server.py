import flwr as fl
import numpy as np
import pickle
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from typing import List, Tuple, Dict
import tensorflow as tf
from flwr.common import Parameters, FitRes

# Load validation data for server-side evaluation
data = pd.read_csv("./UNSW_2018_IoT_Botnet_Dataset_74.csv")
validation_fraction = 0.2
train_data, valid_data = np.split(data.sample(frac=1, random_state=42), [int((1 - validation_fraction) * len(data))])

# Prepare validation data
X_valid = valid_data.drop(columns=['T']).values
y_valid = valid_data['T'].values


# Define Flower server strategy with custom evaluation
class CustomFedAvg(fl.server.strategy.FedAvg):
    def aggregate_fit(
        self, rnd: int, results: List[Tuple[fl.client.Client, FitRes]], failures: List[Tuple[fl.client.Client, Exception]]
    ) -> Tuple[Parameters, Dict[str, float]]:
        """Aggregate weights and calculate server-side metrics."""
        if not results:
            return Parameters(tensors=[], tensor_type=""), {}

        # Extract and deserialize weights
        weights_list = [pickle.loads(res.parameters.tensors[0]) for _, res in results]
        aggregated_weights = self.aggregate(weights_list)

        # Serialize aggregated weights
        serialized_weights = pickle.dumps(aggregated_weights)

        # Evaluate the aggregated model
        r2, rmse = self.evaluate_aggregated_model(aggregated_weights)

        print(f"Round {rnd} - Server Evaluation Metrics: R²: {r2:.4f}, RMSE: {rmse:.4f}")

        return Parameters(
            tensors=[serialized_weights],
            tensor_type="",
        ), {"R2": r2, "RMSE": rmse}

    def aggregate(self, weights_list: List[List[np.ndarray]]) -> List[np.ndarray]:
        """Aggregate model weights using federated averaging."""
        aggregated_weights = []
        num_weights = len(weights_list[0])

        for i in range(num_weights):
            weight_tensors = [weights[i] for weights in weights_list]
            avg_weight = np.mean(weight_tensors, axis=0)
            aggregated_weights.append(avg_weight)

        return aggregated_weights

    def evaluate_aggregated_model(self, aggregated_weights: List[np.ndarray]) -> Tuple[float, float]:
        """Evaluate the aggregated model on the validation data."""
        model = self.create_keras_model((X_valid.shape[1],))
        model.set_weights(aggregated_weights)

        # Predict on validation data
        y_pred = model.predict(X_valid).flatten()

        # Calculate RMSE and R²
        rmse = np.sqrt(mean_squared_error(y_valid, y_pred))
        r2 = r2_score(y_valid, y_pred)

        return r2, rmse

    @staticmethod
    def create_keras_model(input_shape):
        """Create a Keras model."""
        model = tf.keras.models.Sequential([
            tf.keras.layers.Dense(128, activation='relu', input_shape=input_shape),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(1),
        ])
        model.compile(optimizer='adam', loss='mean_squared_error', metrics=['RootMeanSquaredError'])
        return model


# Start Flower server
fl.server.start_server(
    server_address="127.0.0.1:8080",
    strategy=CustomFedAvg(),
)
