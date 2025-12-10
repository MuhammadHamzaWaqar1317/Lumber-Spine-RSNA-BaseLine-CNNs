import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, Input, optimizers, callbacks, mixed_precision
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

# ------------------------------
# Hardware optimization
# ------------------------------
mixed_precision.set_global_policy('mixed_float16')  # Mixed precision for RTX 3090

# ------------------------------
# Dataset & training params
# ------------------------------
DATA_ROOT = r"D:\Machine Learning\Lumber Spine RSNA\Lumber_Spine_RSNA"
OUTPUT_ROOT = r"D:\Machine Learning\Lumber Spine RSNA\RSNA_Training_Results"
IMG_SIZE = (224, 224)
NUM_CLASSES = 3
EPOCHS = 50
BATCH_SIZE = 64
LEVELS = ["L1_L2", "L2_L3", "L3_L4", "L4_L5", "L5_S1"]

# ------------------------------
# Dataset loader
# ------------------------------
def make_ds(level, split='train', augment=False):
    data_dir = os.path.join(DATA_ROOT, level, split)
    ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode='int'
    )
    AUTOTUNE = tf.data.AUTOTUNE
    ds = ds.cache().prefetch(buffer_size=AUTOTUNE)

    if augment:
        augmentation = tf.keras.Sequential([
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.1),
            layers.RandomZoom(0.1)
        ])
        ds = ds.map(lambda x, y: (augmentation(x, training=True), y), num_parallel_calls=AUTOTUNE)
    return ds

# ------------------------------
# HFF_Stem model
# ------------------------------
def conv_bn_relu(x, filters, kernel_size=3, stride=1, name=None):
    x = layers.Conv2D(filters, kernel_size, strides=stride, padding='same',
                      use_bias=False, name=f'{name}_conv')(x)
    x = layers.BatchNormalization(name=f'{name}_bn')(x)
    x = layers.Activation('relu', name=f'{name}_relu')(x)
    return x

def HFF_Stem(input_shape=(224,224,3), output_channels=128):
    inputs = Input(shape=input_shape, name='Input')

    # Stage 1
    l1 = conv_bn_relu(inputs, 32, 3, 1, 'l1')
    l2 = conv_bn_relu(inputs, 32, 5, 1, 'l2')
    f1 = layers.Concatenate(axis=-1, name='concat1')([l1, l2])
    pool1 = layers.MaxPooling2D(2, strides=2, padding='same', name='pool1')(f1)
    l3 = conv_bn_relu(inputs, 32, 5, 2, 'l3')
    f2 = layers.Concatenate(axis=-1, name='concat2')([pool1, l3])
    l4 = conv_bn_relu(inputs, 32, 7, 4, 'l4')

    # Stage 2
    l5 = conv_bn_relu(f2, 32, 3, 1, 'l5')
    l6 = conv_bn_relu(f2, 32, 5, 1, 'l6')
    f3 = layers.Concatenate(axis=-1, name='concat3')([l5, l6])
    pool2 = layers.MaxPooling2D(2, strides=2, padding='same', name='pool2')(f3)
    l7 = conv_bn_relu(f2, 32, 5, 2, 'l7')
    f4 = layers.Concatenate(axis=-1, name='concat4')([pool2, l7, l4])

    # Stage 3
    l8 = conv_bn_relu(f4, 32, 3, 1, 'l8')
    l9 = conv_bn_relu(f4, 32, 5, 1, 'l9')
    f5 = layers.Concatenate(axis=-1, name='concat5')([l8, l9])
    pool3 = layers.MaxPooling2D(2, strides=2, padding='same', name='pool3')(f5)
    l10 = conv_bn_relu(f4, 32, 5, 2, 'l10')
    l11 = conv_bn_relu(f4, 32, 7, 2, 'l11')
    fusion_final = layers.Concatenate(axis=-1, name='concat_final')([pool3, l10, l11])
    output_tensor = conv_bn_relu(fusion_final, output_channels, 1, 1, 'output_proj')
    gap = layers.GlobalAveragePooling2D(name='GAP')(output_tensor)
    gap = layers.Reshape((1,1,output_channels), name='gap_reshape')(gap)

    model = models.Model(inputs=inputs, outputs=gap, name='HFF_Stem')
    return model

# ------------------------------
# Training loop per level
# ------------------------------
for level in LEVELS:
    print(f"\n🔥 Currently Training HFF for Level: {level}")
    
    OUTPUT_DIR = os.path.join(OUTPUT_ROOT, level, "HFF")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load datasets
    train_ds = make_ds(level, "train", augment=True)
    val_ds   = make_ds(level, "val")
    test_ds  = make_ds(level, "test")

    # Instantiate model
    base_model = HFF_Stem()
    model = tf.keras.Sequential([
        base_model,
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.Dense(NUM_CLASSES, activation='softmax', dtype='float32')  # Output float32
    ])

    # Compile
    optimizer = optimizers.Adam(learning_rate=1e-4)
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    # Callbacks
    checkpoint_cb = callbacks.ModelCheckpoint(
        os.path.join(OUTPUT_DIR, "best_model.h5"),
        save_best_only=True,
        monitor='val_accuracy'
    )
    tensorboard_cb = callbacks.TensorBoard(log_dir=os.path.join(OUTPUT_DIR, "logs"))
    earlystop_cb = callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,           # stop if no improvement after 10 epochs
        restore_best_weights=True,
        verbose=1
    )

    # Train
    print(f"📢 Training started for level: {level}")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=[checkpoint_cb, tensorboard_cb, earlystop_cb],
        verbose=2
    )
    print(f"✅ Training finished for level: {level}")

    # ------------------------------
    # Evaluation
    # ------------------------------
    y_pred_probs = model.predict(test_ds, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)

    y_true = []
    for _, labels in test_ds:
        y_true.append(labels.numpy())
    y_true = np.concatenate(y_true)

    class_names = ['Mild', 'Moderate', 'Severe']
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f"{level} Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix.png"), dpi=300, bbox_inches="tight")
    plt.close()

    # Save Classification Report
    report_path = os.path.join(OUTPUT_DIR, "classification_report.txt")
    with open(report_path, "w") as f:
        f.write(classification_report(y_true, y_pred, target_names=class_names))
        f.write(f"\nAccuracy : {accuracy_score(y_true, y_pred):.4f}\n")
        f.write(f"Precision: {precision_score(y_true, y_pred, average='weighted'):.4f}\n")
        f.write(f"Recall   : {recall_score(y_true, y_pred, average='weighted'):.4f}\n")
        f.write(f"F1-score : {f1_score(y_true, y_pred, average='weighted'):.4f}\n")
    
    print(f"✅ Finished training for level {level}. Results saved to {OUTPUT_DIR}")

# ------------------------------
# Print model summary as table
# ------------------------------
model.summary()
