import onnxruntime as ort

def load_model():
    session = ort.InferenceSession(
        "yolov8n.onnx",
        providers=["CPUExecutionProvider"]
    )

    # DEBUG: cek nama input & output
    print("ONNX INPUTS:", [i.name for i in session.get_inputs()])
    print("ONNX OUTPUTS:", [o.name for o in session.get_outputs()])

    return session
