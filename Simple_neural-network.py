import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define the model
def create_model(input_shape=(100, 100, 3), num_classes=33):
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        
        layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    return model

# Data preprocessing
def preprocess_data(train_dir, test_dir, img_size=(100, 100)):
    train_datagen = ImageDataGenerator(rescale=1./255, rotation_range=40, width_shift_range=0.2,
                                      height_shift_range=0.2, shear_range=0.2, zoom_range=0.2,
                                      horizontal_flip=True, fill_mode='nearest')
    
    val_datagen = ImageDataGenerator(rescale=1./255)
    
    train_generator = train_datagen.flow_from_directory(train_dir, target_size=img_size,
                                                       batch_size=32, class_mode='sparse')
    
    val_generator = val_datagen.flow_from_directory(test_dir, target_size=img_size,
                                                   batch_size=32, class_mode='sparse')
    
    return train_generator, val_generator

# Define file paths for dataset (adjust according to your structure)
train_dir = 'Data/train/train'  # Example: './data/train'
test_dir = 'Data/test/test'      # Example: './data/test'

# Load data
train_generator, val_generator = preprocess_data(train_dir, test_dir)

# Create the model
model = create_model()

# Train the model
history = model.fit(train_generator, epochs=10, validation_data=val_generator)

# Save the model
model.save('fruit_classification_model.h5')

# Evaluate the model
test_loss, test_acc = model.evaluate(val_generator)
print(f"Test accuracy: {test_acc:.2f}")
