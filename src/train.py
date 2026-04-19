import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import mlflow
import mlflow.sklearn
import pickle
import os

mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("ride-fare-prediction")

df = pd.read_csv("data/ride_data.csv")

X = df.drop("fare", axis=1)
y = df["fare"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

with mlflow.start_run():

    model = RandomForestRegressor(n_estimators=100)
    model.fit(X_train, y_train)

    score = model.score(X_test, y_test)

    # log everything
    mlflow.log_metric("r2_score", score)
    mlflow.log_param("model_type", "RandomForest")
    mlflow.log_param("test_size", 0.2)
    mlflow.log_param("n_estimators", 100)

    mlflow.sklearn.log_model(model, "model")

    os.makedirs("model", exist_ok=True)
    with open("model/model.pkl", "wb") as f:
        pickle.dump(model, f)

    print("Model trained. R2 Score:", score)