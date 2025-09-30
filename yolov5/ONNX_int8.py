from onnxruntime.quantization import quantize_dynamic, QuantType

quantize_dynamic(
    model_input="best.onnx",
    model_output="best_int8.onnx",
    weight_type=QuantType.QUInt8          # 仅量化权重，激活保持 float
)
print("best_int8.onnx 已生成")