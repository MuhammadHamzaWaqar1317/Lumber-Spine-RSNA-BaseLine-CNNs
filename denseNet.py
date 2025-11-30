import os
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks, applications
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score

# -------------------------------
# CONFIG
# -------------------------------
DATA_ROOT = r"D:\Machine Learning\Lumber_Spine_RSNA\L1_L2"
OUTPUT_DIR = r"D:\Machine Learning\Lumber_Spine_RSNA\models"
IMG_SIZE = (224, 224)
NUM_CLASSES = 3
EPOCHS = 10
SEED = 42
INITIAL_LR = 1e-4

os.makedirs(OUTPUT_DIR, exist_ok=True)

# -------------------------------
# DISTRIBUTED STRATEGY & MIXED PRECISION
# -------------------------------
strategy = tf.distribute.MirroredStrategy()
print("✔ Strategy initialized:", strategy)
tf.keras.mixed_precision.set_global_policy("mixed_float16")
print("✔ Mixed precision enabled.")

# Adjust batch size for full GPU utilization
BATCH_SIZE = 192 * strategy.num_replicas_in_sync  # ~192 for RTX 3090

# -------------------------------
# DATASET PIPELINE
# -------------------------------
def make_ds(folder, augment=False):
    path = os.path.join(DATA_ROOT, folder)
    ds = tf.keras.preprocessing.image_dataset_from_directory(
        path,
        label_mode="int",
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=True if folder=="train" else False,
        seed=SEED
    )

    # Move augmentation to dataset pipeline
    if augment:
        augment_layer = tf.keras.Sequential([
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.07),
            layers.RandomZoom(0.1),
            layers.RandomContrast(0.1)
        ])
        ds = ds.map(lambda x, y: (augment_layer(x, training=True), y), num_parallel_calls=tf.data.AUTOTUNE)

    ds = ds.map(lambda x, y: (applications.densenet.preprocess_input(x), y), num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.cache().prefetch(tf.data.AUTOTUNE)
    return ds

train_ds = make_ds("train", augment=True)
val_ds   = make_ds("val")
test_ds  = make_ds("test")

# -------------------------------
# COMPUTE CLASS WEIGHTS
# -------------------------------
y_train = []
for _, labels in train_ds.unbatch():
    y_train.append(int(labels.numpy()))  # convert scalar tensor to int

y_train = np.array(y_train)
class_weights = compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)
class_weights = {i: float(w) for i, w in enumerate(class_weights)}
print("✔ Class weights:", class_weights)


# -------------------------------
# MODEL BUILDING WITH STRATEGY
# -------------------------------
with strategy.scope():
    inputs = layers.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
    base_model = applications.DenseNet201(include_top=False, weights="imagenet", input_shape=(IMG_SIZE[0], IMG_SIZE[1],3))
    base_model.trainable = False

    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(512, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(NUM_CLASSES, activation="softmax", dtype="float32")(x)  # float32 for mixed precision

    model = models.Model(inputs, outputs)
    model.compile(
        optimizer=optimizers.Adam(INITIAL_LR),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

model.summary()

# -------------------------------
# CALLBACKS
# -------------------------------
checkpoint_path = os.path.join(OUTPUT_DIR, "best_densenet201.h5")
cbs = [
    callbacks.ModelCheckpoint(checkpoint_path, monitor="val_accuracy", save_best_only=True, verbose=1),
    callbacks.EarlyStopping(monitor="val_accuracy", patience=8, restore_best_weights=True, verbose=1),
    callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, verbose=1, min_lr=1e-7)
]

# -------------------------------
# STAGE 1: TRAIN HEAD
# -------------------------------
print("\n===== Stage 1: Train Classifier Head =====\n")
model.fit(train_ds, validation_data=val_ds, epochs=10, class_weight=class_weights, callbacks=cbs)

# -------------------------------
# STAGE 2: FINE-TUNE BASE
# -------------------------------
print("\n===== Stage 2: Fine-tuning DenseNet201 =====\n")
with strategy.scope():
    base_model.trainable = True
    fine_tune_at = 300
    for i, layer in enumerate(base_model.layers):
        if i < fine_tune_at:
            layer.trainable = False
    model.compile(
        optimizer=optimizers.Adam(1e-5),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, initial_epoch=10, class_weight=class_weights, callbacks=cbs)

# -------------------------------
# FINAL EVALUATION
# -------------------------------
print("\n===== Final Test Evaluation =====\n")
results = model.evaluate(test_ds)
print("Test results:", results)
model.save(os.path.join(OUTPUT_DIR, "final_densenet201.h5"))
print("✔ Model saved.")

# -------------------------------
# FAST TEST METRICS
# -------------------------------
print("\n===== FAST TEST REPORT =====\n")

# 1. Predict from test_ds (GPU optimized)
y_pred_probs = model.predict(test_ds, verbose=0)
y_pred = np.argmax(y_pred_probs, axis=1)

# 2. Very fast label extraction
print("Extracting labels (FAST MODE)...")
y_true = []
for _, labels in test_ds:
    y_true.append(labels.numpy())
y_true = np.concatenate(y_true)

class_names = ['Mild', 'Moderate', 'Severe']

# 3. Confusion Matrix
cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# 4. Metrics
print(f"Accuracy : {accuracy_score(y_true, y_pred):.4f}")
print(f"Precision: {precision_score(y_true, y_pred, average='weighted'):.4f}")
print(f"Recall   : {recall_score(y_true, y_pred, average='weighted'):.4f}")
print(f"F1-Score : {f1_score(y_true, y_pred, average='weighted'):.4f}")

print("\n", classification_report(y_true, y_pred, target_names=class_names))
