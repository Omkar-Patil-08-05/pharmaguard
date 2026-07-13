#data_preprocessing.py
import pandas as pd
import os

# ==========================================================
# Load Dataset
# ==========================================================

file_path = "dataset/DataCoSupplyChainDataset.csv"

df = pd.read_csv(file_path, encoding="latin1")

print("=" * 60)
print("ORIGINAL DATASET")
print("=" * 60)
print("Shape:", df.shape)

# ==========================================================
# Feature Extraction
# Select only logistics-related features
# ==========================================================

selected_columns = [
    "Days for shipping (real)",
    "Days for shipment (scheduled)",
    "Delivery Status",
    "Late_delivery_risk",
    "Customer City",
    "Customer Country",
    "Order City",
    "Order Country",
    "order date (DateOrders)",
    "shipping date (DateOrders)",
    "Order Item Quantity",
    "Shipping Mode",
    "Order Status"
]

df = df[selected_columns]

print("\nFeature Extraction Completed.")
print("Selected Features:", len(df.columns))

# ==========================================================
# Rename Columns
# ==========================================================

df.rename(columns={
    "Days for shipping (real)": "transit_time",
    "Days for shipment (scheduled)": "scheduled_transit_time",
    "Delivery Status": "delivery_status",
    "Late_delivery_risk": "late_delivery_risk",
    "Customer City": "destination_city",
    "Customer Country": "destination_country",
    "Order City": "source_city",
    "Order Country": "source_country",
    "order date (DateOrders)": "shipment_date",
    "shipping date (DateOrders)": "delivery_date",
    "Order Item Quantity": "quantity",
    "Shipping Mode": "shipping_mode",
    "Order Status": "order_status"
}, inplace=True)

# ==========================================================
# Missing Values
# ==========================================================

print("\n" + "=" * 60)
print("MISSING VALUES BEFORE CLEANING")
print("=" * 60)
print(df.isnull().sum())

df.dropna(inplace=True)

print("\n" + "=" * 60)
print("MISSING VALUES AFTER CLEANING")
print("=" * 60)
print(df.isnull().sum())

# ==========================================================
# Duplicate Analysis
# ==========================================================

duplicates = df.duplicated().sum()

print("\n" + "=" * 60)
print("DUPLICATE RECORDS")
print("=" * 60)
print("Duplicate Rows:", duplicates)

# ----------------------------------------------------------
# NOTE:
# We DO NOT remove duplicates because multiple shipments
# can legitimately have identical logistics information.
# Keeping them preserves the complete shipment history.
# ----------------------------------------------------------

# ==========================================================
# Dataset Information
# ==========================================================

print("\n" + "=" * 60)
print("FINAL DATASET")
print("=" * 60)

print("Shape:", df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nFirst Five Rows:")
print(df.head())

# ==========================================================
# Save Clean Dataset
# ==========================================================

os.makedirs("dataset", exist_ok=True)

output_path = "dataset/cleaned_data.csv"

df.to_csv(output_path, index=False)

print("\nCleaned dataset saved successfully.")
print("Location:", output_path)