import pandas as pd
import os

os.makedirs("data", exist_ok=True)

print("Loading dataset...")

df = pd.read_csv("data/uber.csv")

print("Original shape:", df.shape)

# drop missing values
df = df.dropna()

# convert datetime
df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"])

# extract hour
df["hour"] = df["pickup_datetime"].dt.hour

# compute distance
df["distance_km"] = (
    abs(df["pickup_latitude"] - df["dropoff_latitude"]) +
    abs(df["pickup_longitude"] - df["dropoff_longitude"])
) * 111

# remove outliers
df = df[df["distance_km"] < 50]
df = df[df["fare_amount"] < 100]

# feature engineering
df["distance_hour_interaction"] = df["distance_km"] * df["hour"]

# select columns
df = df[["distance_km", "hour", "distance_hour_interaction", "fare_amount"]]
df.rename(columns={"fare_amount": "fare"}, inplace=True)

# reduce dataset size
df = df.sample(1500, random_state=42)

# save
df.to_csv("data/ride_data.csv", index=False)

print("Processed dataset saved:", df.shape)