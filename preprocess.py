import pandas as pd
import os

# ensure folder exists
os.makedirs("data", exist_ok=True)

# load dataset
df = pd.read_csv("data/uber.csv")

print("Original shape:", df.shape)

# drop missing values
df = df.dropna()

# convert datetime
df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"])

# extract hour
df["hour"] = df["pickup_datetime"].dt.hour

# calculate distance (simple approximation)
df["distance_km"] = (
    abs(df["pickup_latitude"] - df["dropoff_latitude"]) +
    abs(df["pickup_longitude"] - df["dropoff_longitude"])
) * 111

# keep only useful columns
df = df[["distance_km", "hour", "fare_amount"]]
df.rename(columns={"fare_amount": "fare"}, inplace=True)

# reduce size (VERY IMPORTANT for DVC)
df = df.sample(2000, random_state=42)

# save cleaned dataset
df.to_csv("data/ride_data.csv", index=False)

print("Processed dataset saved:", df.shape)