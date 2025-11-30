import os
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks, applications
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score

# ------------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------------
DATA_ROOT = r"D:\Machine Learning\Lumber_Spine_RSNA"
OUTPUT_ROOT = r"D:\Machine Learning\RSNA_Training_Results"
IMG_SIZE = (224, 224)  # Xception expects 299x299 by default, but we can resize
NUM_CLASSES = 3
EPOCHS = 50
INITIAL_LR = 1e-4
SEED = 42

LEVELS = ["L1_L2", "L2_L3", "L3_L4", "L4_L5", "L5_S1"]

# MIRRORED STRATEGY + MIXED PRECISION
strategy = tf.distribute.MirroredStrategy()
tf.keras.mixed_precision.set_global_policy("mixed_float16")

BATCH_SIZE = 192 * strategy.num_replicas_in_sync

# ------------------------------------------------------------------
# DATASET PIPELINE
# ------------------------------------------------------------------
def make_ds(level, folder, augment=False):
    path = os.path.join(DATA_ROOT, level, folder)

    ds = tf.keras.preprocessing.image_dataset_from_directory(
        path,
        label_mode="int",
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=True if folder == "train" else False,
        seed=SEED
    )

    if augment:
        augment_layer = tf.keras.Sequential([
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.07),
            layers.RandomZoom(0.1),
            layers.RandomContrast(0.1)
        ])
        ds = ds.map(lambda x, y: (augment_layer(x, training=True), y),
                    num_parallel_calls=tf.data.AUTOTUNE)

    ds = ds.map(lambda x, y: (applications.xception.preprocess_input(x), y),
                num_parallel_calls=tf.data.AUTOTUNE)

    return ds.cache().prefetch(tf.data.AUTOTUNE)

# ------------------------------------------------------------------
# TRAINING LOOP FOR EACH SPINE LEVEL
# ------------------------------------------------------------------
for level in LEVELS:
    print("\n" + "="*70)
    print(f"🔥 Training Xception for Level: {level}")
    print("="*70)

    # FINAL OUTPUT STRUCTURE
    OUTPUT_DIR = os.path.join(OUTPUT_ROOT, level, "Xception")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Datasets
    train_ds = make_ds(level, "train", augment=True)
    val_ds   = make_ds(level, "val")
    test_ds  = make_ds(level, "test")

    # Class Weights
    y_train = []
    for _, labels in train_ds.unbatch():
        y_train.append(int(labels.numpy()))
    class_weights = {
        i: float(w)
        for i, w in enumerate(compute_class_weight("balanced",
                                                   classes=np.unique(y_train),
                                                   y=y_train))
    }

    # MODEL BUILD
    with strategy.scope():
        inputs = layers.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
        base_model = applications.Xception(include_top=False, weights="imagenet",
                                          input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
        base_model.trainable = False

        x = base_model(inputs, training=False)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dropout(0.4)(x)
        x = layers.Dense(512, activation="relu")(x)
        x = layers.Dropout(0.3)(x)
        outputs = layers.Dense(NUM_CLASSES, activation="softmax", dtype="float32")(x)

        model = models.Model(inputs, outputs)
        model.compile(optimizer=optimizers.Adam(INITIAL_LR),
                      loss="sparse_categorical_crossentropy",
                      metrics=["accuracy"])

    # CALLBACKS
    checkpoint_path = os.path.join(OUTPUT_DIR, "best_model.h5")
    cbs = [
        callbacks.ModelCheckpoint(checkpoint_path, monitor="val_accuracy",
                                  save_best_only=True, verbose=1),
        callbacks.EarlyStopping(monitor="val_accuracy", patience=12,
                                restore_best_weights=True, verbose=1),
        callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                    patience=3, verbose=1, min_lr=1e-7)
    ]

    # ---------------------------------------------------------------
    # TRAIN HEAD
    # ---------------------------------------------------------------
    print("\n===== Stage 1: Train Classifier Head =====")
    model.fit(train_ds, validation_data=val_ds, epochs=10,
              class_weight=class_weights, callbacks=cbs)

    # ---------------------------------------------------------------
    # FINE TUNE
    # ---------------------------------------------------------------
    print("\n===== Stage 2: Fine Tuning Base Model =====")
    with strategy.scope():
        base_model.trainable = True
        fine_tune_at = 100  # Xception has fewer layers than DenseNet, so adjust
        for i, layer in enumerate(base_model.layers):
            if i < fine_tune_at:
                layer.trainable = False

        model.compile(
            optimizer=optimizers.Adam(1e-5),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"]
        )

    model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS,
              initial_epoch=10, class_weight=class_weights, callbacks=cbs)

    # ---------------------------------------------------------------
    # FINAL TEST
    # ---------------------------------------------------------------
    print("\n===== Final Evaluation =====")
    results = model.evaluate(test_ds)
    print("Test Results:", results)

    # Save final full model
    model.save(os.path.join(OUTPUT_DIR, "final_model.h5"))

    # ---------------------------------------------------------------
    # METRICS + CONFUSION MATRIX + REPORT
    # ---------------------------------------------------------------
    y_pred_probs = model.predict(test_ds, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)

    # Extract y_true
    y_true = []
    for _, labels in test_ds:
        y_true.append(labels.numpy())
    y_true = np.concatenate(y_true)

    class_names = ['Mild', 'Moderate', 'Severe']

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f"{level} Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    cm_path = os.path.join(OUTPUT_DIR, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=300, bbox_inches="tight")
    plt.close()

    # Calculate TP, FP, FN, TN for each class
    tp_fp_fn_tn = []
    for i in range(len(class_names)):
        TP = cm[i, i]
        FP = cm[:, i].sum() - TP
        FN = cm[i, :].sum() - TP
        TN = cm.sum() - (TP + FP + FN)
        tp_fp_fn_tn.append((TP, FP, FN, TN))

    # Save Classification Report with TP, FP, FN, TN
    report_path = os.path.join(OUTPUT_DIR, "classification_report.txt")
    with open(report_path, "w") as f:
        f.write(classification_report(y_true, y_pred, target_names=class_names))
        f.write("\n")
        f.write(f"Accuracy : {accuracy_score(y_true, y_pred):.4f}\n")
        f.write(f"Precision: {precision_score(y_true, y_pred, average='weighted'):.4f}\n")
        f.write(f"Recall   : {recall_score(y_true, y_pred, average='weighted'):.4f}\n")
        f.write(f"F1-score: {f1_score(y_true, y_pred, average='weighted'):.4f}\n\n")
        f.write("Class-wise TP, FP, FN, TN:\n")
        for idx, cls in enumerate(class_names):
            TP, FP, FN, TN = tp_fp_fn_tn[idx]
            f.write(f"{cls}: TP={TP}, FP={FP}, FN={FN}, TN={TN}\n")
