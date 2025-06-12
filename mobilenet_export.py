import tensorflow as tf
import tf2onnx
import onnx

# MobileNet Model Creation
def create_mobilenet():
    from tensorflow.keras.applications import MobileNet
    mobilenet_base = MobileNet(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    
    # Freeze layers except for the last 5 layers
    for layer in mobilenet_base.layers[:-5]:
        layer.trainable = False
    
    mobilenet = tf.keras.Sequential([
        mobilenet_base,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(5, activation='softmax')
    ])
    
    mobilenet.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])  # Ensure it's compiled
    return mobilenet

# Convert MobileNet model to ONNX
def export_mobilenet_to_onnx():
    mobilenet_model = create_mobilenet()
    
    # Save the model as a SavedModel
    mobilenet_model.save("mobilenet_saved_model")
    print("Model saved as SavedModel")
    
    # Load the SavedModel
    loaded_model = tf.keras.models.load_model("mobilenet_saved_model")
    
    # Convert the SavedModel to ONNX format using tf2onnx
    import tf2onnx
    import onnx
    
    # Convert the model to ONNX using tf2onnx's from_keras method
    onnx_model = tf2onnx.convert.from_keras(loaded_model, opset=13)
    
    # Save the ONNX model to disk
    onnx.save_model(onnx_model, "mobilenet.onnx")
    
    print("✅ MobileNet model saved as ONNX!")

# Run the export function
export_mobilenet_to_onnx()
