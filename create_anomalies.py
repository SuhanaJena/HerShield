import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest

INPUT_FILE = "data/processed/hershield_risk_features_2023.csv"
OUTPUT_FILE = "data/processed/hershield_anomalies_2023.csv"

df = pd.read_csv(INPUT_FILE)

anomaly_features = [
    "total_rate",
    "rape_rate",
    "dowry_rate",
    "cruelty_rate",
    "kidnapping_rate",
    "assault_rate",
    "cyber_rate",
    "pocso_rate",
]

X = df[anomaly_features]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = IsolationForest(
    contamination=0.10,
    random_state=42,
    n_estimators=100
)

df["anomaly_prediction"] = model.fit_predict(X_scaled)
df["anomaly_score"] = model.decision_function(X_scaled)

df["anomaly_status"] = df["anomaly_prediction"].map({
    1: "Normal Profile",
    -1: "Anomalous Profile"
})

print("Isolation Forest anomaly detection completed!")
print("Output file:", OUTPUT_FILE)

print("\nAnomaly counts:")
print(
    df["anomaly_status"]
    .value_counts()
    .to_string()
)

print("\nAnomaly states:")
print(
    df[
        [
            "state",
            "risk_indicator",
            "risk_category",
            "anomaly_score",
            "anomaly_status"
        ]
    ]
    .sort_values("anomaly_score")
    .to_string(index=False)
)

df.to_csv(OUTPUT_FILE, index=False)

print("\nSaved:", OUTPUT_FILE)
print("Shape:", df.shape)