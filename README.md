# PharmaGuard
### AI-Powered Pharmaceutical Supply Chain Fraud Detection using Machine Learning

---

# Project Overview

PharmaGuard is an AI-powered pharmaceutical supply chain monitoring system designed to identify suspicious shipment behavior before medicines reach distributors or pharmacies.

The current version focuses on building an industry-standard Machine Learning pipeline capable of learning normal shipment behavior from historical logistics data and assigning a fraud risk score to every shipment.

The system is designed such that the trained ML model can later be integrated with:

- Blockchain-based traceability
- FastAPI backend
- React Dashboard
- QR Verification
- Real-time IoT sensors (Future Phase)

---

# Problem Statement

Counterfeit medicines and supply chain fraud are major challenges in the pharmaceutical industry.

Some common problems include:

- Counterfeit medicine insertion
- Cold-chain violations
- Unexpected warehouse stops
- Shipment delays
- Unauthorized shipment modifications
- Tampered logistics records

Traditional logistics systems only record shipment events but do not proactively identify suspicious shipment behavior.

PharmaGuard aims to solve this problem by learning shipment patterns using Machine Learning and assigning a risk score to every shipment.

---

# Current Project Scope

Current implementation focuses entirely on software.

No hardware is used.

Sensor values are currently simulated or obtained from historical datasets.

Future versions will replace simulated values with real IoT sensor data without changing the AI pipeline.

---

# Technology Stack

## Programming Language

- Python 3.12

## Machine Learning

- Scikit-Learn
- NumPy
- Pandas
- Joblib

## Visualization

- Matplotlib

## Future Technologies

- FastAPI
- React
- PostgreSQL
- Polygon Blockchain
- Docker

---

# Dataset

Dataset Used

DataCo Supply Chain Dataset

Dataset Size

180,519 Records

Current ML Features

10 Features

---

# Machine Learning Pipeline

```
Online Dataset
        │
        ▼
Data Cleaning
        │
        ▼
Feature Engineering
        │
        ▼
Exploratory Data Analysis
        │
        ▼
Preprocessing
        │
        ▼
Processed Dataset
        │
        ▼
Model Training
        │
        ▼
Risk Score Generation
        │
        ▼
Model Evaluation
```

---

# Current ML Workflow

```
Raw Dataset
      │
      ▼
Data Exploration
      │
      ▼
Dataset Validation
      │
      ▼
Feature Engineering
      │
      ▼
EDA
      │
      ▼
Preprocessing
      │
      ▼
Isolation Forest
      │
      ▼
Risk Score
```

---

# Project Structure

```
PHARMAGUARD

│
├── ai
│   ├── explore_dataset.py
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   └── dataset_validation.py
│
├── dataset
│
├── docs
│
├── ml
│   │
│   ├── data
│   │
│   ├── eda
│   │
│   ├── preprocessing
│   │
│   ├── models
│   │
│   ├── evaluation
│   │
│   ├── inference
│   │
│   ├── saved_models
│   │
│   └── utils
│
├── main.py
│
└── README.md
```

---

# Features Used for Machine Learning

The first version of the ML model intentionally uses behavioural shipment features instead of identifiers.

## Numerical Features

- transit_time
- scheduled_transit_time
- distance
- temperature
- warehouse_stops
- quantity
- late_delivery_risk

## Encoded Categorical Features

- shipping_mode
- delivery_status
- order_status

Total Features

10

---

# Removed Features

The following columns were intentionally excluded from model training.

```
batch_id
shipment_id
manufacturer
distributor
source_city
destination_city
source_country
destination_country
shipment_date
delivery_date
```

Reason:

These are identifiers or metadata and may reduce the ability of the model to generalize to new manufacturers or locations.

---

# Exploratory Data Analysis

Completed

✔ Dataset Information

✔ Summary Statistics

✔ Missing Value Analysis

✔ Duplicate Analysis

✔ Unique Value Analysis

✔ Constant Column Detection

✔ Data Type Analysis

✔ Numerical Histograms

Generated Reports

```
summary_statistics.csv

missing_values.csv

duplicate_report.csv

unique_values.csv

constant_columns.csv

data_types.csv

data_quality_report.md
```

---

# Data Preprocessing

The preprocessing pipeline performs

- Identifier removal
- Date conversion
- Feature selection
- Ordinal Encoding
- Standard Scaling
- Pipeline Serialization

Output

```
processed_dataset.csv

preprocessing_pipeline.pkl
```

---

# Final Processed Dataset

Rows

180,519

Columns

10

All features are numerical.

Ready for Machine Learning.

---

# First Machine Learning Model

Isolation Forest

Purpose

Unsupervised anomaly detection.

Reason for choosing

- No fraud labels available
- Fast
- Industry standard
- Good baseline model

Configuration

```
IsolationForest(

n_estimators=300,

contamination=0.02,

random_state=42,

max_samples="auto",

n_jobs=-1

)
```

---

# Current Model Performance

Dataset

180,519 Rows

Training Time

5.402 Seconds

Inference Time

8.362 Seconds

Anomaly Percentage

2%

Normal Shipments

176,910

Anomalies

3,609

Average Risk Score

27.73

Risk Levels

Low

116,173

Medium

55,984

High

8,362

---

# Risk Score Generation

Isolation Forest generates anomaly scores.

These scores are converted into

0 – 100

using Min-Max Normalization.

Higher Risk Score

↓

Higher Fraud Probability

Risk Levels

Low

0–30

Medium

31–70

High

71–100

---

# Outputs Generated

Model

```
saved_models/

isolation_forest.pkl
```

Prediction File

```
docs/reports/

isolation_forest_predictions.csv
```

Summary Report

```
docs/reports/

isolation_forest_summary.md
```

---

# Current Development Status

| Module | Status |
|---------|--------|
| Dataset Collection | ✅ |
| Data Cleaning | ✅ |
| Feature Engineering | ✅ |
| Dataset Validation | ✅ |
| Exploratory Data Analysis | ✅ |
| Preprocessing | ✅ |
| Isolation Forest | ✅ |
| Risk Score Generation | ✅ |
| Model Saving | ✅ |
| Prediction Generation | ✅ |

---

# Upcoming Development

The following anomaly detection models will be implemented and compared.

- Local Outlier Factor (LOF)
- One-Class SVM
- Elliptic Envelope
- Autoencoder

After benchmarking all models, the best-performing model will be selected for deployment.

---

# Future Roadmap

```
Machine Learning
        │
        ▼
Best Model Selection
        │
        ▼
FastAPI Backend
        │
        ▼
Blockchain Integration
        │
        ▼
PostgreSQL
        │
        ▼
Shipment Simulator
        │
        ▼
React Dashboard
        │
        ▼
QR Verification
        │
        ▼
Real-Time IoT Integration
```

---

# Future Hardware Integration

Current Version

Software Only

Future Version

- Temperature Sensors
- GPS Modules
- ESP32
- Raspberry Pi
- Warehouse IoT Gateway

The software architecture has been designed so that hardware can be integrated later without modifying the Machine Learning pipeline.

---

# Authors

PES University

Industry-Level Capstone Project

PharmaGuard