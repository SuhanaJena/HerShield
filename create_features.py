import pandas as pd

input_file = "data/processed/hershield_crime_2023_clean.csv"
output_file = "data/processed/hershield_features_2023.csv"

df = pd.read_csv(input_file)

new_columns = [
    "state",
    "total_cases",
    "total_victims",
    "total_rate",
    "ipc_cases",
    "ipc_victims",
    "ipc_rate",
    "sll_cases",
    "sll_victims",
    "sll_rate",
    "rape_cases",
    "rape_victims",
    "rape_rate",
    "dowry_deaths",
    "dowry_victims",
    "dowry_rate",
    "cruelty_cases",
    "cruelty_victims",
    "cruelty_rate",
    "kidnapping_cases",
    "kidnapping_victims",
    "kidnapping_rate",
    "acid_attack_cases",
    "acid_attack_victims",
    "acid_attack_rate",
    "assault_cases",
    "assault_victims",
    "assault_rate",
    "trafficking_cases",
    "trafficking_victims",
    "trafficking_rate",
    "cyber_cases",
    "cyber_victims",
    "cyber_rate",
    "pocso_cases",
    "pocso_victims",
    "pocso_rate",
]

df.columns = new_columns

df.to_csv(output_file, index=False)

print("Feature engineering completed!")
print("Created:", output_file)
print("Shape:", df.shape)

print("\nColumns:")
for i, column in enumerate(df.columns, start=1):
    print(f"{i}. {column}")

print("\nFirst 5 rows:")
print(df.head().to_string(index=False))