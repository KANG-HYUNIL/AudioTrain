"""
Export model to ONNX format for interoperability.
"""

def export_to_onnx(model, dummy_input, onnx_path: str = "model.onnx"):
    # TODO: torch.onnx.export with dynamic axes
    return onnx_path
