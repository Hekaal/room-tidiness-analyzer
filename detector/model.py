import onnxruntime as ort

def load_model():
    session = ort.InferenceSession(
        "yolov8n.onnx",
        providers=["CPUExecutionProvider"]
    )
    return session
