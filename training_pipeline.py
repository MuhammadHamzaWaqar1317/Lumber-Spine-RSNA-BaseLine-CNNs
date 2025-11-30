import tensorflow as tf
import os
import matplotlib.pyplot as plt

# ------------------------------
# GPU MIXED PRECISION (3090)
# ------------------------------
tf.keras.mixed_precision.set_global_policy("mixed_float16")

# ------------------------------
# CONFIGURATION
# ------------------------------
DATA_DIR = r"D:\Machine Learning\Lumber Spine RSNA Sagittal T1,T2 Cleaned Data"
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 50

# ------------------------------
# LOAD DATASET
# ------------------------------
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    DATA_DIR,
    labels="inferred",
    label_mode="categorical",
    validation_split=0.2,
    subset="training",
    seed=42,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE
)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    DATA_DIR,
    labels="inferred",
    label_mode="categorical",
    validation_split=0.2,
    subset="validation",
    seed=42,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE
)

class_names = train_ds.class_names
print("Classes:", class_names)

# ------------------------------
# DATA AUGMENTATION
# ------------------------------
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.10),
    tf.keras.layers.RandomZoom(0.15),
])

# ------------------------------
# NORMALIZATION + PREFETCH
# ------------------------------
normalization = tf.keras.layers.Rescaling(1./255)

train_ds = train_ds.map(lambda x, y: (normalization(x), y))
val_ds = val_ds.map(lambda x, y: (normalization(x), y))

train_ds = train_ds.cache().prefetch(tf.data.AUTOTUNE)
val_ds = val_ds.cache().prefetch(tf.data.AUTOTUNE)

# ------------------------------
# BUILD MODEL (Functional API)
# ------------------------------

# Input
inputs = tf.keras.Input(shape=(224, 224, 3))

# Augmentation (runs only on training)
x = data_augmentation(inputs)

# Base model
base_model = tf.keras.applications.DenseNet201(
    include_top=False,
    weights="imagenet",
    input_shape=(224, 224, 3)
)
base_model.trainable = True

x = base_model(x)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dropout(0.4)(x)
x = tf.keras.layers.Dense(512, activation="relu")(x)
x = tf.keras.layers.Dropout(0.3)(x)

# Output logits must be float32 with mixed precision
outputs = tf.keras.layers.Dense(
    len(class_names),
    activation="softmax",
    dtype="float32"  # very important for mixed precision!
)(x)

model = tf.keras.Model(inputs, outputs)

model.summary()

# ------------------------------
# CALLBACKS
# ------------------------------
callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        "best_densenet201_rsna.h5",
        monitor="val_accuracy",
        save_best_only=True
    ),
    tf.keras.callbacks.EarlyStopping(
        monitor="val_accuracy",
        patience=7,
        restore_best_weights=True
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.2,
        patience=4,
        verbose=1
    )
]

# ------------------------------
# COMPILE
# ------------------------------
model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-4),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# ------------------------------
# TRAIN
# ------------------------------
history = model.fit(
    train_ds,
    epochs=EPOCHS,
    validation_data=val_ds,
    callbacks=callbacks
)

model.save("final_densenet201_rsna.h5")
print("\nModel saved as final_densenet201_rsna.h5")

# ------------------------------
# PLOT TRAINING CURVES
# ------------------------------
plt.figure(figsize=(12, 6))

# Accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history["accuracy"], label="Train Accuracy")
plt.plot(history.history["val_accuracy"], label="Val Accuracy")
plt.legend()
plt.title("Accuracy")

# Loss
plt.subplot(1, 2, 2)
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Val Loss")
plt.legend()
plt.title("Loss")

plt.savefig("training_plots.png")
plt.show()
