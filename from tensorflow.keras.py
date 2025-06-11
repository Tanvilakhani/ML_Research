from tensorflow.keras.models import load_model

# Load the trained model
model = load_model('fruit_classification_model.h5')

# Print model summary (optional)
model.summary()
