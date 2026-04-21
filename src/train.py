import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import mlflow
import mlflow.sklearn
import pickle
import os

mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("ride-fare-prediction")

# load data
df = pd.read_csv("data/ride_data.csv")

X = df.drop("fare", axis=1)
y = df["fare"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# define models
models = {
    "LinearRegression": LinearRegression(),
    "DecisionTree": DecisionTreeRegressor(),
    "RandomForest": RandomForestRegressor(n_estimators=100)
}

best_model = None
best_score = -999
best_name = ""

# train + compare
for name, model in models.items():

    with mlflow.start_run(run_name=name):

        model.fit(X_train, y_train)
        score = model.score(X_test, y_test)

        mlflow.log_param("model_name", name)
        mlflow.log_metric("r2_score", score)

        mlflow.sklearn.log_model(model, name)

        print(f"{name} R2 Score: {score}")

        if score > best_score:
            best_score = score
            best_model = model
            best_name = name

# save best model
os.makedirs("model", exist_ok=True)

with open("model/model.pkl", "wb") as f:
    pickle.dump(best_model, f)

print(f"\nBest Model: {best_name} with R2 Score: {best_score}")