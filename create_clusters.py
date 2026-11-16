import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

INPUT_FILE = "data/processed/hershield_risk_features_2023.csv"
OUTPUT_FILE = "data/processed/hershield_clusters_2023.csv"

df = pd.read_csv(INPUT_FILE)

cluster_features = [
    "total_rate",
    "rape_rate",
    "dowry_rate",
    "cruelty_rate",
    "kidnapping_rate",
    "assault_rate",
    "cyber_rate",
    "pocso_rate",
]

X = df[cluster_features]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

kmeans = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10
)

df["cluster"] = kmeans.fit_predict(X_scaled)

cluster_summary = (
    df.groupby("cluster")[cluster_features]
    .mean()
    .round(2)
)

print("K-Means clustering completed!")
print("Number of clusters:", kmeans.n_clusters)
print("Output file:", OUTPUT_FILE)

print("\nCluster sizes:")
print(df["cluster"].value_counts().sort_index().to_string())

print("\nCluster profiles:")
print(cluster_summary.to_string())

print("\nState cluster assignments:")
print(
    df[
        ["state", "risk_indicator", "risk_category", "cluster"]
    ]
    .sort_values(["cluster", "risk_indicator"], ascending=[True, False])
    .to_string(index=False)
)

df.to_csv(OUTPUT_FILE, index=False)

print("\nSaved:", OUTPUT_FILE)
print("Shape:", df.shape)