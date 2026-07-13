# feature_engineering.py
import pandas as pd
import numpy as np
import random
import os

# ==========================================================
# Load Cleaned Dataset
# ==========================================================

file_path = "dataset/cleaned_data.csv"

df = pd.read_csv(file_path)

print("=" * 60)
print("CLEANED DATASET LOADED")
print("=" * 60)
print(df.shape)

# ==========================================================
# Generate Batch ID
# ==========================================================

df["batch_id"] = [
    f"BATCH_{i:06d}"
    for i in range(1, len(df) + 1)
]

# ==========================================================
# Generate Shipment ID
# ==========================================================

df["shipment_id"] = [
    f"SHIP_{i:06d}"
    for i in range(1, len(df) + 1)
]

# ==========================================================
# Manufacturer List
# ==========================================================

manufacturers = [
    "Sun Pharma",
    "Cipla",
    "Dr. Reddy's",
    "Lupin",
    "Aurobindo Pharma",
    "Torrent Pharma",
    "Alkem Labs",
    "Mankind Pharma"
]

df["manufacturer"] = np.random.choice(
    manufacturers,
    size=len(df)
)

# ==========================================================
# Distributor List
# ==========================================================

distributors = [
    "MedSupply Logistics",
    "HealthChain Distribution",
    "PharmaTrans",
    "CareLogistics",
    "MediExpress",
    "SafeDrug Distribution"
]

df["distributor"] = np.random.choice(
    distributors,
    size=len(df)
)

# ==========================================================
# Temperature Generation
# Based on Shipping Mode
# ==========================================================

def generate_temperature(mode):

    if mode == "Same Day":
        return round(np.random.normal(18, 2), 1)

    elif mode == "First Class":
        return round(np.random.normal(22, 2), 1)

    elif mode == "Second Class":
        return round(np.random.normal(24, 2), 1)

    else:
        return round(np.random.normal(26, 2), 1)

df["temperature"] = df["shipping_mode"].apply(generate_temperature)

# ==========================================================
# Warehouse Stops
# Based on Transit Time
# ==========================================================

def warehouse_stops(days):

    if days <= 2:
        return 1

    elif days <= 5:
        return 2

    elif days <= 7:
        return 3

    else:
        return 4

df["warehouse_stops"] = df["transit_time"].apply(warehouse_stops)

# ==========================================================
# Distance Estimation
# Correlated with Transit Time
# ==========================================================

distance = []

for transit in df["transit_time"]:

    base = transit * 250

    noise = random.randint(-80, 120)

    distance.append(max(50, base + noise))

df["distance"] = distance

# ==========================================================
# Reorder Columns
# ==========================================================

column_order = [
    "batch_id",
    "shipment_id",
    "manufacturer",
    "distributor",
    "source_city",
    "destination_city",
    "source_country",
    "destination_country",
    "shipment_date",
    "delivery_date",
    "transit_time",
    "scheduled_transit_time",
    "distance",
    "temperature",
    "warehouse_stops",
    "quantity",
    "shipping_mode",
    "delivery_status",
    "late_delivery_risk",
    "order_status"
]

df = df[column_order]

# ==========================================================
# Save Engineered Dataset
# ==========================================================

os.makedirs("dataset", exist_ok=True)

df.to_csv(
    "dataset/engineered_data.csv",
    index=False
)

print("\nFeature Engineering Completed Successfully!")

print("\nFinal Dataset Shape:")
print(df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nFirst Five Rows:")
print(df.head())