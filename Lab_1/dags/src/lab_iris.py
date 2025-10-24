import os
import pickle

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

def load_iris_data():
    """
    Loads the Iris dataset and returns serialized data.
    """
    iris = load_iris()
    X, y = iris.data, iris.target
    serialized_data = pickle.dumps((X, y))
    return serialized_data


def preprocess_data(data):
    """
    Deserializes Iris data, scales features, splits into train/test, and returns serialized data.
    """
    X, y = pickle.loads(data)

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )
    serialized_train_test = pickle.dumps((X_train, X_test, y_train, y_test))
    return serialized_train_test


def train_mlp(data, hidden_dim=10):
    """
    Trains an MLPClassifier on Iris data and saves the model.
    """
    X_train, X_test, y_train, y_test = pickle.loads(data)

    mlp = MLPClassifier(hidden_layer_sizes=(hidden_dim,), max_iter=500, random_state=42)
    mlp.fit(X_train, y_train)

    # Save model
    model_dir = os.path.join(os.path.dirname(__file__), "../model")
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "mlp_iris.pkl")
    with open(model_path, 'wb') as f:
        pickle.dump(mlp, f)

    # Return test data for evaluation
    return pickle.dumps((X_test, y_test))


def evaluate_mlp(data):
    """
    Loads trained MLP and evaluates accuracy on the test set.
    """
    X_test, y_test = pickle.loads(data)

    model_path = os.path.join(os.path.dirname(__file__), "../model/mlp_iris.pkl")
    mlp = pickle.load(open(model_path, 'rb'))

    y_pred = mlp.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"MLP Accuracy on Iris test set: {acc:.4f}")
