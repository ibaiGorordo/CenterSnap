import onnx
import onnxoptimizer

model_path = "CenterSnap.onnx"

onnx_model = onnx.load(model_path)
passes = ["extract_constant_to_initializer", "eliminate_unused_initializer"]
optimized_model = onnxoptimizer.optimize(onnx_model, passes)

onnx.save(optimized_model, model_path)