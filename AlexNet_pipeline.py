import os

# ==============================
# Must set these BEFORE importing tensorflow
# ==============================
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices=false'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'   # optional: limit to one GPU
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'  # avoid full VRAM allocation

# ==============================
# Now import TensorFlow
# ==============================
import tensorflow as tf
import numpy as np
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.metrics import (
    confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score
)
import matplotlib.pyplot as plt
import seaborn as sns

# GPU memory growth
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"[INFO] {len(gpus)} GPU(s) detected. Memory growth enabled.")
    except RuntimeError as e:
        print(e)


# -----------------------------
# AlexNet Base
# -----------------------------
def lrn(x):
    return tf.nn.local_response_normalization(x, depth_radius=2, bias=1.0, alpha=1e-4, beta=0.75)

def AlexNet_base(input_shape=(227,227,3)):
    inp = layers.Input(shape=input_shape)
    x = layers.Conv2D(96, (11, 11), strides=4, activation='relu')(inp)
    x = layers.Lambda(lrn)(x)
    x = layers.MaxPool2D((3, 3), strides=2)(x)

    x = layers.Conv2D(256, (5,5), padding='same', activation='relu', groups=2)(x)
    x = layers.Lambda(lrn)(x)
    x = layers.MaxPool2D((3,3), strides=2)(x)

    x = layers.Conv2D(384, (3,3), padding='same', activation='relu')(x)
    x = layers.Conv2D(384, (3,3), padding='same', activation='relu', groups=2)(x)
    x = layers.Conv2D(256, (3,3), padding='same', activation='relu', groups=2)(x)
    x = layers.MaxPool2D((3,3), strides=2)(x)
    return models.Model(inputs=inp, outputs=x, name="AlexNet_base")

def AlexNet(num_classes=3, input_shape=(227,227,3)):
    base = AlexNet_base(input_shape)
    x = layers.Flatten()(base.output)
    x = layers.Dense(4096, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(4096, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    return models.Model(inputs=base.input, outputs=outputs, name="AlexNet")

# -----------------------------
# General Parameters
# -----------------------------
BASE_DIR = r"D:\Machine Learning\Lumber_Spine_RSNA"
LEVELS = ["L1_L2", "L2_L3", "L3_L4", "L4_L5", "L5_S1"]
img_size = (227,227)
batch_size = 32
epochs = 10
num_classes = 3  # Mild, Moderate, Severe
class_names = ["Mild", "Moderate", "Severe"]

# -----------------------------
# Loop over levels
# -----------------------------
for level in LEVELS:
    print(f"\n===== TRAINING LEVEL: {level} =====\n")
    
    train_dir = os.path.join(BASE_DIR, level, "train")
    val_dir   = os.path.join(BASE_DIR, level, "val")
    test_dir  = os.path.join(BASE_DIR, level, "test")
    
    OUTPUT_DIR = os.path.join(BASE_DIR, level, "results")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Datasets
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        train_dir, image_size=img_size, batch_size=batch_size, label_mode="categorical"
    )
    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        val_dir, image_size=img_size, batch_size=batch_size, label_mode="categorical"
    )
    test_ds = tf.keras.preprocessing.image_dataset_from_directory(
        test_dir, image_size=img_size, batch_size=1, label_mode="categorical", shuffle=False
    )
    
    # Model
    model = AlexNet(num_classes=num_classes, input_shape=(227,227,3))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    
    # Checkpoint
    best_model_path = os.path.join(OUTPUT_DIR, "alexnet_best.keras")
    checkpoint = ModelCheckpoint(best_model_path, monitor="val_accuracy", save_best_only=True, verbose=1)
    
    # Train
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=[checkpoint],
        verbose=1
    )
    
    # -----------------------------
    # Evaluation & Metrics
    # -----------------------------
    model = tf.keras.models.load_model(best_model_path)
    
    y_pred_probs = model.predict(test_ds, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)

    y_true = []
    for _, labels in test_ds:
        y_true.append(labels.numpy())
    y_true = np.concatenate(y_true)
    y_true = np.argmax(y_true, axis=1)

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.title(f"{level} Confusion Matrix")
    plt.xlabel("Predicted"); plt.ylabel("Actual")
    plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix.png"), dpi=300, bbox_inches="tight")
    plt.close()

    # Compute TP, FP, FN, TN
    tp_fp_fn_tn = []
    for i in range(len(class_names)):
        TP = cm[i,i]
        FP = cm[:,i].sum() - TP
        FN = cm[i,:].sum() - TP
        TN = cm.sum() - (TP + FP + FN)
        tp_fp_fn_tn.append((TP, FP, FN, TN))
    
    # Save classification report
    with open(os.path.join(OUTPUT_DIR, "classification_report.txt"), "w") as f:
        f.write(classification_report(y_true, y_pred, target_names=class_names))
        f.write("\n")
        f.write(f"Accuracy : {accuracy_score(y_true, y_pred):.4f}\n")
        f.write(f"Precision: {precision_score(y_true, y_pred, average='weighted'):.4f}\n")
        f.write(f"Recall   : {recall_score(y_true, y_pred, average='weighted'):.4f}\n")
        f.write(f"F1-score : {f1_score(y_true, y_pred, average='weighted'):.4f}\n\n")
        f.write("Class-wise TP, FP, FN, TN:\n")
        for idx, cls in enumerate(class_names):
            TP, FP, FN, TN = tp_fp_fn_tn[idx]
            f.write(f"{cls}: TP={TP}, FP={FP}, FN={FN}, TN={TN}\n")
    
    print(f"\n===== FINISHED LEVEL: {level} =====\n")
