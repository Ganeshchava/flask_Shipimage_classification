import os
import tensorflow as tf
import tf2onnx

# Ensure the models folder exists
os.makedirs("models", exist_ok=True)

# Function to create VGG16 model
def create_vgg16():
    from tensorflow.keras.applications import VGG16
    base = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    for layer in base.layers[:-4]:
        layer.trainable = False
    model = tf.keras.Sequential([
        base,
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(5, activation='softmax')
    ])
    return model

# Function to create ResNet50 model
def create_resnet50():
    from tensorflow.keras.applications import ResNet50
    base = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    for layer in base.layers[:-10]:
        layer.trainable = False
    model = tf.keras.Sequential([
        base,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(5, activation='softmax')
    ])
    return model

# Function to create InceptionV3 model
def create_inception():
    from tensorflow.keras.applications import InceptionV3
    base = InceptionV3(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    for layer in base.layers[:-10]:
        layer.trainable = False
    model = tf.keras.Sequential([
        base,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(5, activation='softmax')
    ])
    return model

# Function to create MobileNet model
def create_mobilenet():
    from tensorflow.keras.applications import MobileNet
    base = MobileNet(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    for layer in base.layers[:-5]:
        layer.trainable = False
    model = tf.keras.Sequential([
        base,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(5, activation='softmax')
    ])
    return model

# Function to create custom CNN model
def create_custom_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3), padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2, 2)),

        tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2, 2)),

        tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2, 2)),

        tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2, 2)),

        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(5, activation='softmax')
    ])
    return model

# Function to export any model to ONNX
def export_to_onnx(model, model_name):
    spec = (tf.TensorSpec((None, 224, 224, 3), tf.float32, name="input"),)
    onnx_model, _ = tf2onnx.convert.from_keras(model, input_signature=spec, opset=13)
    with open(f"models/{model_name}.onnx", "wb") as f:
        f.write(onnx_model.SerializeToString())
    print(f"✅ Saved: {model_name}.onnx")

# === Build and export all models ===
vgg16_model = create_vgg16()
export_to_onnx(vgg16_model, "vgg16")

resnet_model = create_resnet50()
export_to_onnx(resnet_model, "resnet50")

inception_model = create_inception()
export_to_onnx(inception_model, "inception")

mobilenet_model = create_mobilenet()
export_to_onnx(mobilenet_model, "mobilenet")

custom_model = create_custom_model()
export_to_onnx(custom_model, "custom_model")
