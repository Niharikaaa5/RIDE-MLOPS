from fastapi import FastAPI
import pickle
import logging

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

        prediction = model.predict([[distance, hour]])
        result = round(float(prediction[0]),2)

        # log request
        logging.info(f"{data} -> {result}")

        return {"predicted_fare": result}

    except Exception as e:
        return {"error": str(e)}