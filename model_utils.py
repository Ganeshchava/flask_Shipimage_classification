# model_utils.py

import onnxruntime as rt
import numpy as np
from PIL import Image

class_names = ['Cargo', 'Military', 'Carrier', 'Cruise', 'Tanker']

# Load models once
sessions = {
    'vgg': rt.InferenceSession("models/vgg16.onnx"),
    'resnet': rt.InferenceSession("models/resnet50.onnx"),
    'inception': rt.InferenceSession("models/inceptionv3.onnx"),
    'custom': rt.InferenceSession("models/custom_cnn.onnx")
}

def preprocess_image(image):
    image = image.resize((224, 224)).convert('RGB')
    img = np.array(image).astype(np.float32) / 255.0
    return np.expand_dims(img, axis=0)

def predict_ensemble(image):
    input_name = sessions['vgg'].get_inputs()[0].name
    img = preprocess_image(image)

    predictions = [sess.run(None, {input_name: img})[0] for sess in sessions.values()]
    avg_prediction = np.mean(predictions, axis=0)

    predicted_class = np.argmax(avg_prediction)
    confidence = float(np.max(avg_prediction))

    return class_names[predicted_class], confidence
