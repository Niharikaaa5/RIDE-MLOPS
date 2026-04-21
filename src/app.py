from fastapi import FastAPI
import pickle
import logging
import os

# ensure logs folder exists
os.makedirs("logs", exist_ok=True)

# logging setup
logging.basicConfig(
    filename="logs/logs.txt",
    level=logging.INFO,
    format="%(asctime)s - %(message)s"
)

app = FastAPI()

# load trained model
model = pickle.load(open("model/model.pkl", "rb"))

@app.get("/")
def home():
    return {"message": "API is running"}

@app.post("/predict")
def predict(data: dict):
    try:
        distance = data["distance_km"]
        hour = data["hour"]

        # ✅ compute new feature internally
        interaction = distance * hour

        # ✅ now matches training features
        prediction = model.predict([[distance, hour, interaction]])
        result = round(float(prediction[0]), 2)

        # log request
        logging.info(f"{data} -> {result}")

        return {"predicted_fare": result}

    except Exception as e:
        return {"error": str(e)}