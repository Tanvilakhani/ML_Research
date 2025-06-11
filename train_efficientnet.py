import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report

# Parameters
IMG_SIZE = 224  # EfficientNetB0 default input size
BATCH_SIZE = 32
EPOCHS = 10
NUM_CLASSES = 10
TRAIN_DIR = r'C:\Users\admin\Documents\ML_research\MY_data\train'

# Load the pretrained model without top layers
base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))

# Add custom top layers
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation='relu')(x)
predictions = Dense(NUM_CLASSES, activation='softmax')(x)

# Create the full model
model = Model(inputs=base_model.input, outputs=predictions)

# Freeze the base EfficientNetB0 layers
for layer in base_model.layers:
    layer.trainable = False

# Compile the model
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Data augmentation for training
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    validation_split=0.2
)

# Load training data
train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training'
)

# Load validation data
validation_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation'
)

# Train the model
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // BATCH_SIZE,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // BATCH_SIZE,
    epochs=EPOCHS
)

# Fine-tune the model: unfreeze some layers
for layer in model.layers[-20:]:
    layer.trainable = True

# Recompile with a lower learning rate
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Continue training
history_fine = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // BATCH_SIZE,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // BATCH_SIZE,
    epochs=5
)

# ✅ Save full model in HDF5 format
model.save('efficientnetb0_fruits_model.h5')
print("Model saved successfully as HDF5 (.h5)")

# Save the model in different formats
try:
    # Save complete model in SavedModel format
    model.save('efficientnetb0_fruits_model')
    print("Model saved in SavedModel format")
    
    # Save model architecture and weights in H5 format
    tf.keras.models.save_model(
        model,
        'efficientnetb0_fruits_model.h5',
        include_optimizer=True,
        save_format='h5',
        save_traces=False
    )
    print("Model saved in H5 format")
    
    # Save just the weights separately
    model.save_weights('efficientnetb0_fruits_weights.h5')
    print("Model weights saved separately")

except Exception as e:
    print(f"Error saving model: {str(e)}")
    # Fallback to saving only weights
    model.save_weights('efficientnetb0_fruits_weights.h5')
    print("Fallback: Only weights were saved")

# Print class indices
print("\nClass indices:")
print(train_generator.class_indices)

# Combine training history
total_acc = history.history['accuracy'] + history_fine.history['accuracy']
total_val_acc = history.history['val_accuracy'] + history_fine.history['val_accuracy']
total_loss = history.history['loss'] + history_fine.history['loss']
total_val_loss = history.history['val_loss'] + history_fine.history['val_loss']

# Plot training metrics
epochs_range = range(1, len(total_acc) + 1)
plt.figure(figsize=(15, 5))

# Plot accuracy
plt.subplot(1, 2, 1)
plt.plot(epochs_range, total_acc, label='Training Accuracy')
plt.plot(epochs_range, total_val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('EfficientNetB0 Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')

# Plot loss
plt.subplot(1, 2, 2)
plt.plot(epochs_range, total_loss, label='Training Loss')
plt.plot(epochs_range, total_val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('EfficientNetB0 Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')

plt.tight_layout()
plt.savefig('efficientnetb0_training_metrics.png')
plt.show()

# Final metrics
print("\nFinal Training Metrics:")
print(f"Training Accuracy: {total_acc[-1]:.4f}")
print(f"Validation Accuracy: {total_val_acc[-1]:.4f}")
print(f"Training Loss: {total_loss[-1]:.4f}")
print(f"Validation Loss: {total_val_loss[-1]:.4f}")

# Per-class performance
validation_generator.reset()
predictions = model.predict(validation_generator)
y_pred = np.argmax(predictions, axis=1)
y_true = validation_generator.classes

print("\nPer-class Performance:")
print(classification_report(
    y_true,
    y_pred,
    target_names=list(train_generator.class_indices.keys())
))
