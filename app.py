import os
import numpy as np
import onnxruntime as ort
from flask import Flask, render_template, request
from PIL import Image

app = Flask(__name__)

# Define upload folder and model path
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# ONNX model paths for different models
# model_paths = {
#     'resnet50': 'C:/ship_class/models/resnet50.onnx',
#     'vgg16': 'C:/ship_class/models/vgg16.onnx',
#     'inceptionv3': 'C:/ship_class/models/inceptionv3.onnx',
#     'mobilenet': 'C:/ship_class/models/mobilenet.onnx'
# }
model_paths = {
    "vgg16": "models/vgg16.onnx",
    "resnet50": "models/resnet50.onnx"
}


# Initialize ONNX sessions for each model
sessions = {model_name: ort.InferenceSession(model_paths[model_name]) for model_name in model_paths}

# Define class names for prediction
class_names = ['Cargo', 'Military', 'Carrier', 'Cruise', 'Tanker']

# def preprocess_image(file, model_name="resnet50"):
#     """
#     Preprocesses the image for the given model.
#     This includes resizing the image based on the model input size, normalization, and ensuring the correct dimensions.
#     """
#     img = Image.open(file).convert("RGB")

#     # Resize the image based on the model's expected input size
#     if model_name in ["resnet50", "vgg16"]:
#         img = img.resize((224, 224))  # Expected input size for ResNet50 and VGG16
#     elif model_name in ["inceptionv3", "mobilenet"]:
#         img = img.resize((299, 299))  # Expected input size for InceptionV3 and MobileNet

#     img = np.array(img).astype(np.float32)  # Convert image to float32 for processing

#     # Normalize image to [0, 1]
#     img = img / 255.0

#     # Convert to a batch of images (add an extra dimension at the start)
#     img = np.expand_dims(img, axis=0)

#     # Change channel order from HWC to CHW for ONNX models
#     img = np.transpose(img, (0, 3, 1, 2))  # (batch_size, channels, height, width)

#     return img
def preprocess_image(file, model_name="resnet50"):
    img = Image.open(file).convert("RGB")

    if model_name in ["resnet50", "vgg16"]:
        img = img.resize((224, 224))
    elif model_name in ["inceptionv3", "mobilenet"]:
        img = img.resize((299, 299))

    img = np.array(img).astype(np.float32) / 255.0  # Normalize to [0, 1]

    img = np.expand_dims(img, axis=0)  # Shape: [1, 224, 224, 3]

    return img


@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':
        if 'image' not in request.files:
            return render_template('index.html', prediction='No file uploaded.')

        file = request.files['image']
        model_name = request.form.get('model', 'resnet50')  # Get the selected model

        if file.filename == '':
            return render_template('index.html', prediction='No selected file.')

        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)

        try:
            # Preprocess image based on selected model
            img_array = preprocess_image(file_path, model_name=model_name)
            
            # Get the ONNX session for the selected model
            session = sessions[model_name]
            
            # Get input name and perform inference
            input_name = session.get_inputs()[0].name
            outputs = session.run(None, {input_name: img_array})
            
            # Get prediction and display class name
            pred_class = np.argmax(outputs[0])
            prediction = class_names[pred_class]
        except Exception as e:
            prediction = f"Error during prediction: {e}"

    return render_template('index.html', prediction=prediction)


if __name__ == '__main__':
    app.run(debug=True)
