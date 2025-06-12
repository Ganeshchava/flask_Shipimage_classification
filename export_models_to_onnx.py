from convert_to_onnx import (
    create_vgg16, create_resnet50, create_inception,
    create_mobilenet, create_custom_model, export_to_onnx
)

def export_all_models():
    models = {
        "vgg16": create_vgg16(),
        "resnet50": create_resnet50(),
        "inception": create_inception(),
        "mobilenet": create_mobilenet(),
        "custom_model": create_custom_model()
    }

    for name, model in models.items():
        export_to_onnx(model, name)

if __name__ == "__main__":
    export_all_models()
