import onnxruntime as ort

# Specify the full path to your ONNX model file
model_path = r"C:\ship_class\model\resnet50.onnx"

# Print the path to make sure it's correct
print(f"Model Path: {model_path}")

# Try to load the ONNX model using ONNX Runtime
try:
    session = ort.InferenceSession(model_path)
    print("Model loaded successfully")
except Exception as e:
    print(f"Error: {e}")
