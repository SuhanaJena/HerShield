import os
import numpy as np
import pandas as pd
import tensorflow as tf

from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

BASE = "data/real_world_evaluation/external/UofT-CIFAR-Synthetic-Evidence-Image-Corpus-fda969f"

SOURCE_DIR = os.path.join(BASE, "Source-Images")
MANIP_DIR = os.path.join(BASE, "Manipulated-Images")
MANIFEST = os.path.join(BASE, "image_corpus_manifest.csv")

MODEL_PATH = "models/image_authenticity_model.keras"
OUTPUT_DIR = "data/real_world_evaluation/results"
OUTPUT_CSV = os.path.join(OUTPUT_DIR, "real_world_cnn_results.csv")

THRESHOLD = 0.5

print("\nLoading manifest...")
df = pd.read_csv(MANIFEST)
test_df = df[df["pool"] == "test"].copy().reset_index(drop=True)

print("Test pairs:", len(test_df))

print("\nLoading existing HerShield CNN...")
model = tf.keras.models.load_model(MODEL_PATH, compile=False)

print("Model loaded successfully.")
print("Input shape:", model.input_shape)
print("Output shape:", model.output_shape)

images = []
records = []

print("\nPreparing original and manipulated images...")

for index, row in test_df.iterrows():
    frame_id = str(row["frame_id"])
    source_path = os.path.join(SOURCE_DIR, frame_id + ".png")

    manipulated_name = str(row["assigned_item_id"])
    manipulated_path = os.path.join(MANIP_DIR, manipulated_name)

    for image_path, image_type in [
        (source_path, "REAL"),
        (manipulated_path, "FAKE")
    ]:
        with Image.open(image_path) as image:
            image_rgb = image.convert("RGB")
            image_rgb = image_rgb.resize(
                (32, 32),
                Image.Resampling.LANCZOS
            )
            image_array = np.asarray(
                image_rgb,
                dtype=np.float32
            )

        images.append(image_array)

        records.append({
            "frame_id": frame_id,
            "image_type": image_type,
            "family": row["family"],
            "tier": row["tier"],
            "assigned_variant": row["assigned_variant"],
            "image_filename": os.path.basename(image_path)
        })

print("Images prepared:", len(images))

X = np.stack(images, axis=0)

print("\nRunning CNN predictions...")
predictions = model.predict(X, batch_size=32, verbose=1)

predictions = np.asarray(predictions).reshape(-1)

if len(predictions) != len(records):
    raise ValueError("Unexpected number of model predictions.")

if not np.all(np.isfinite(predictions)):
    raise ValueError("Model returned invalid prediction values.")

if np.any(predictions < 0) or np.any(predictions > 1):
    raise ValueError(
        "Expected sigmoid probabilities between 0 and 1. "
        "Check the model output before continuing."
    )

results = pd.DataFrame(records)

results["probability_real"] = predictions
results["fake_class_score"] = 1.0 - predictions

results["predicted_class"] = np.where(
    results["fake_class_score"] >= THRESHOLD,
    "FAKE",
    "REAL"
)

results["correct"] = (
    results["image_type"] == results["predicted_class"]
)

# Save detailed predictions.
os.makedirs(OUTPUT_DIR, exist_ok=True)
results.to_csv(OUTPUT_CSV, index=False)

print("\n" + "=" * 60)
print("HERShield REAL-WORLD GENERALISATION TEST")
print("=" * 60)

y_true = results["image_type"]
y_pred = results["predicted_class"]

print("\n--- OVERALL RESULTS ---")
print("Images evaluated:", len(results))
print("Authentic source images:", sum(y_true == "REAL"))
print("Manipulated images:", sum(y_true == "FAKE"))

print(f"Accuracy:  {accuracy_score(y_true, y_pred) * 100:.2f}%")
print(f"Precision: {precision_score(y_true, y_pred, pos_label='FAKE', zero_division=0) * 100:.2f}%")
print(f"Recall:    {recall_score(y_true, y_pred, pos_label='FAKE', zero_division=0) * 100:.2f}%")
print(f"F1-score:  {f1_score(y_true, y_pred, pos_label='FAKE', zero_division=0) * 100:.2f}%")

print("\n--- CONFUSION MATRIX ---")
print("Rows = Actual | Columns = Predicted")
print("Order: REAL, FAKE")

cm = confusion_matrix(
    y_true,
    y_pred,
    labels=["REAL", "FAKE"]
)

print(pd.DataFrame(
    cm,
    index=["Actual REAL", "Actual FAKE"],
    columns=["Predicted REAL", "Predicted FAKE"]
).to_string())

print("\n--- CLASSIFICATION REPORT ---")
print(classification_report(
    y_true,
    y_pred,
    labels=["REAL", "FAKE"],
    zero_division=0
))

print("\n--- RESULTS BY IMAGE FAMILY ---")

for family in sorted(results["family"].unique()):
    subset = results[results["family"] == family]

    print(
        f"{family}: "
        f"{accuracy_score(subset['image_type'], subset['predicted_class']) * 100:.2f}% "
        f"accuracy ({len(subset)} images)"
    )

print("\n--- RESULTS BY MANIPULATION TIER ---")

for tier in sorted(test_df["tier"].unique()):
    frame_ids = test_df.loc[test_df["tier"] == tier, "frame_id"]

    subset = results[results["frame_id"].isin(frame_ids)]

    print(
        f"{tier}: "
        f"{accuracy_score(subset['image_type'], subset['predicted_class']) * 100:.2f}% "
        f"accuracy ({len(subset)} images)"
    )

print("\n--- PAIRED IMAGE ANALYSIS ---")

wide = results.pivot(
    index="frame_id",
    columns="image_type",
    values="predicted_class"
)

correct_pairs = (
    (wide["REAL"] == "REAL") &
    (wide["FAKE"] == "FAKE")
).sum()

both_real = (
    (wide["REAL"] == "REAL") &
    (wide["FAKE"] == "REAL")
).sum()

both_fake = (
    (wide["REAL"] == "FAKE") &
    (wide["FAKE"] == "FAKE")
).sum()

reverse_pairs = (
    (wide["REAL"] == "FAKE") &
    (wide["FAKE"] == "REAL")
).sum()

print("Correctly distinguished pairs:", correct_pairs, "/", len(wide))
print("Both classified REAL-like:", both_real)
print("Both classified FAKE-like:", both_fake)
print("Reverse classification:", reverse_pairs)

print("\n--- OUTPUT FILE ---")
print(OUTPUT_CSV)

print("\nEvaluation completed.")
print(
    "\nIMPORTANT: This is an experimental generalisation test "
    "on a separate image-manipulation benchmark. It is not "
    "proof of real-world crime-photo authenticity."
)
