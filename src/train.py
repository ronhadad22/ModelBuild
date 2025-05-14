import argparse
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from joblib import dump

def train_model(model_path):
    data = load_iris()
    X, y = data.data, data.target
    model = RandomForestClassifier()
    model.fit(X, y)
    dump(model, model_path)
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="model.joblib")
    args = parser.parse_args()
    train_model(args.model_path)
