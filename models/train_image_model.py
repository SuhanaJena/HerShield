
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Dataset paths
TRAIN_DIR = "data/external/cifake/train"
TEST_DIR = "data/external/cifake/test"
MODEL_PATH = "models/image_authenticity_model.keras"

IMAGE_SIZE = (32, 32)
BATCH_SIZE = 64
EPOCHS = 5
SEED = 42

# Load training data (90% training, 10% validation)
print("\nLoading training dataset...")

train_ds = keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    class_names=["FAKE", "REAL"],
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="binary",
    validation_split=0.1,
    subset="training",
    seed=SEED
)

val_ds = keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    class_names=["FAKE", "REAL"],
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="binary",
    validation_split=0.1,
    subset="validation",
    seed=SEED
)

# Load independent test data
print("\nLoading test dataset...")

test_ds = keras.utils.image_dataset_from_directory(
    TEST_DIR,
    class_names=["FAKE", "REAL"],
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="binary",
    shuffle=False
)

print("\nClass mapping: FAKE = 0, REAL = 1")

# Improve data loading performance
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)
test_ds = test_ds.prefetch(buffer_size=AUTOTUNE)

# Build CNN architecture
print("\nBuilding CNN model...")

model = keras.Sequential([
    layers.Input(shape=(32, 32, 3)),
    layers.Rescaling(1.0 / 255),

    layers.Conv2D(32, 3, activation="relu", padding="same"),
    layers.MaxPooling2D(),

    layers.Conv2D(64, 3, activation="relu", padding="same"),
    layers.MaxPooling2D(),

    layers.Conv2D(128, 3, activation="relu", padding="same"),
    layers.MaxPooling2D(),

    layers.GlobalAveragePooling2D(),
    layers.Dense(64, activation="relu"),
    layers.Dropout(0.4),
    layers.Dense(1, activation="sigmoid")
])

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=[
        "accuracy",
        keras.metrics.Precision(name="precision"),
        keras.metrics.Recall(name="recall")
    ]
)

model.summary()

# Stop early if validation performance stops improving
early_stopping = keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=2,
    restore_best_weights=True
)

# Train
print("\nStarting CNN training...")

model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=[early_stopping]
)

# Evaluate on independent test dataset
print("\nEvaluating model...")

results = model.evaluate(test_ds, return_dict=True)

for metric, value in results.items():
    print(f"{metric}: {value:.4f}")

# Save model
os.makedirs("models", exist_ok=True)
model.save(MODEL_PATH)

print("\nModel saved successfully!")
print("Saved at:", MODEL_PATH)
print("Output probability represents the REAL class.")
print("AI-generated screening score = 1 - REAL probability.")

