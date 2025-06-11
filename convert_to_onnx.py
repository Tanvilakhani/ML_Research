
import tf2onnx
import tensorflow as tf
import onnx

# Load the saved Keras model
model = tf.keras.models.load_model('resnet50_fruits_trained.h5')

# Convert the model to ONNX
# Define input shape (batch_size, height, width, channels)
spec = (tf.TensorSpec((None, 224, 224, 3), tf.float32, name="input"),)

# Convert model to ONNX format
model_proto, _ = tf2onnx.convert.from_keras(model, input_signature=spec, opset=13)

# Save the ONNX model
output_path = "resnet50_fruits.onnx"
onnx.save(model_proto, output_path)

print(f"Model converted and saved to {output_path}")

# Verify the model
onnx_model = onnx.load(output_path)
onnx.checker.check_model(onnx_model)
print("ONNX model checked - conversion successful!")


## convert MobileNetV2 model to ONNX format
import tf2onnx
import tensorflow as tf
import onnx

# Load the saved MobileNetV2 model
model = tf.keras.models.load_model('mobilenetv2_fruits_trained.h5')

# Convert the model to ONNX
# Define input shape (batch_size, height, width, channels)
spec = (tf.TensorSpec((None, 224, 224, 3), tf.float32, name="input"),)

# Convert model to ONNX format
model_proto, _ = tf2onnx.convert.from_keras(model, input_signature=spec, opset=13)

# Save the ONNX model
output_path = "mobilenetv2_fruits.onnx"
onnx.save(model_proto, output_path)

print(f"Model converted and saved to {output_path}")

# Verify the model
onnx_model = onnx.load(output_path)
onnx.checker.check_model(onnx_model)
print("ONNX model checked - conversion successful!")