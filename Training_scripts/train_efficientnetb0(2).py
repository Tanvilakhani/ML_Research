import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import pandas as pd
from tensorflow.keras.applications.efficientnet import EfficientNetB0, preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import mixed_precision

# -----------------------
# Save Paths (Local)
# -----------------------
BASE_DIR = 'efficientnet_project'
MODEL_PATH = os.path.join(BASE_DIR, 'models/efficientnet_b0_fruits_trained.keras')
CSV_PATH = os.path.join(BASE_DIR, 'results/report/efficientnet_b0_training_history.csv')
REPORT_PATH = os.path.join(BASE_DIR, 'results/report/efficientnet_b0_classification_report.txt')
METRICS_PLOT_PATH = os.path.join(BASE_DIR, 'results/metrics/efficientnet_b0_training_metrics.png')
CONF_MATRIX_PATH = os.path.join(BASE_DIR, 'results/metrics/efficientnet_b0_confusion_matrix.png')
CHECKPOINT_DIR = os.path.join(BASE_DIR, 'checkpoints_efficientnet_b0')

# Create directories
for path in [MODEL_PATH, CSV_PATH, REPORT_PATH, METRICS_PLOT_PATH, CONF_MATRIX_PATH]:
    os.makedirs(os.path.dirname(path), exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# -----------------------
# Threading and Precision
# -----------------------
tf.config.threading.set_intra_op_parallelism_threads(6)
tf.config.threading.set_inter_op_parallelism_threads(6)
mixed_precision.set_global_policy('float32')

# -----------------------
# Constants
# -----------------------
IMG_SIZE = 224   # EfficientNetB0 default input size
BATCH_SIZE = 16
EPOCHS_INITIAL = 10
EPOCHS_FINE = 5
NUM_CLASSES = 10
TRAIN_DIR = 'MY_data/train'  # <-- Update this with your local dataset path

# -----------------------
# Build EfficientNetB0 Model
# -----------------------
base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation='relu')(x)
predictions = Dense(NUM_CLASSES, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=predictions)

# Freeze base model for initial training
for layer in base_model.layers:
    layer.trainable = False

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# -----------------------
# Data Generators
# -----------------------
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    validation_split=0.2
)

train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    color_mode='rgb',
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training'
)

validation_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    color_mode='rgb',
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation',
    shuffle=False
)

# -----------------------
# Checkpoint Callback
# -----------------------
checkpoint_callback = ModelCheckpoint(
    filepath=os.path.join(CHECKPOINT_DIR, 'efficientnet_b0_best.keras'),
    monitor='val_accuracy',
    save_best_only=True,
    verbose=1
)

# -----------------------
# Initial Training
# -----------------------
history = model.fit(
    train_generator,
    steps_per_epoch=int(np.ceil(train_generator.samples / BATCH_SIZE)),
    validation_data=validation_generator,
    validation_steps=int(np.ceil(validation_generator.samples / BATCH_SIZE)),
    epochs=EPOCHS_INITIAL,
    callbacks=[checkpoint_callback]
)

# -----------------------
# Fine-Tuning
# -----------------------
for layer in base_model.layers[-80:]:
    layer.trainable = True

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

history_fine = model.fit(
    train_generator,
    steps_per_epoch=int(np.ceil(train_generator.samples / BATCH_SIZE)),
    validation_data=validation_generator,
    validation_steps=int(np.ceil(validation_generator.samples / BATCH_SIZE)),
    epochs=EPOCHS_FINE,
    callbacks=[checkpoint_callback]
)

# -----------------------
# Save Final Model
# -----------------------
model.save(MODEL_PATH)

# -----------------------
# Class Mapping
# -----------------------
class_indices = train_generator.class_indices
idx_to_label = {v: k for k, v in class_indices.items()}

# -----------------------
# Combine History
# -----------------------
total_acc = history.history['accuracy'] + history_fine.history['accuracy']
total_val_acc = history.history['val_accuracy'] + history_fine.history['val_accuracy']
total_loss = history.history['loss'] + history_fine.history['loss']
total_val_loss = history.history['val_loss'] + history_fine.history['val_loss']
epochs_range = range(1, len(total_acc) + 1)

# Save history as CSV
history_df = pd.DataFrame({
    'epoch': list(epochs_range),
    'train_acc': total_acc,
    'val_acc': total_val_acc,
    'train_loss': total_loss,
    'val_loss': total_val_loss
})
history_df.to_csv(CSV_PATH, index=False)

# -----------------------
# Plot and Save Training Metrics
# -----------------------
plt.figure(figsize=(15, 5))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, total_acc, label='Training Accuracy')
plt.plot(epochs_range, total_val_acc, label='Validation Accuracy')
plt.legend()
plt.title('Accuracy')
plt.subplot(1, 2, 2)
plt.plot(epochs_range, total_loss, label='Training Loss')
plt.plot(epochs_range, total_val_loss, label='Validation Loss')
plt.legend()
plt.title('Loss')
plt.tight_layout()
plt.savefig(METRICS_PLOT_PATH)
plt.close()

# -----------------------
# Evaluation
# -----------------------
validation_generator.reset()
predictions = model.predict(validation_generator)
y_pred = np.argmax(predictions, axis=1)
y_true = validation_generator.classes

# Classification Report
report = classification_report(y_true, y_pred, target_names=list(class_indices.keys()))
print(report)
with open(REPORT_PATH, 'w') as f:
    f.write(report)

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(class_indices.keys()))
fig, ax = plt.subplots(figsize=(10, 8))
disp.plot(ax=ax, cmap='Blues', xticks_rotation=45, colorbar=False)
plt.tight_layout()
plt.savefig(CONF_MATRIX_PATH)
plt.close()
