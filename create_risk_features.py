import pandas as pd
from sklearn.preprocessing import MinMaxScaler

INPUT_FILE = "data/processed/hershield_features_2023.csv"
OUTPUT_FILE = "data/processed/hershield_risk_features_2023.csv"

df = pd.read_csv(INPUT_FILE)

risk_features = [
    "total_rate",
    "rape_rate",
    "dowry_rate",
    "cruelty_rate",
    "kidnapping_rate",
    "assault_rate",
    "cyber_rate",
    "pocso_rate",
]

scaler = MinMaxScaler()

scaled = scaler.fit_transform(df[risk_features])

scaled_df = pd.DataFrame(
    scaled,
    columns=[f"{column}_normalized" for column in risk_features]
)

df = pd.concat([df, scaled_df], axis=1)

normalized_columns = scaled_df.columns.tolist()

df["risk_indicator"] = (
    df[normalized_columns].mean(axis=1) * 100
)

df["risk_indicator"] = df["risk_indicator"].round(2)


def classify_risk(score):
    if score < 33.33:
        return "Lower Indicator"
    elif score < 66.67:
        return "Moderate Indicator"
    else:
        return "Higher Indicator"


df["risk_category"] = df["risk_indicator"].apply(classify_risk)

df.to_csv(OUTPUT_FILE, index=False)

print("Risk feature engineering completed!")
print("Created:", OUTPUT_FILE)
print("Shape:", df.shape)

print("\nRisk indicator summary:")
print(df["risk_indicator"].describe().round(2).to_string())

print("\nRisk category counts:")
print(df["risk_category"].value_counts().to_string())

print("\nState risk indicators:")
print(
    df[
        ["state", "risk_indicator", "risk_category"]
    ]
    .sort_values("risk_indicator", ascending=False)
    .to_string(index=False)
)