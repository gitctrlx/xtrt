import onnx

model_path = "./xtrt/weights/yolov5s_ccpd.onnx" 
model = onnx.load(model_path)

old_new_names = {
    "DecodeNumDetection": "num_dets",
    "DecodeDetectionBoxes": "boxes",
    "DecodeDetectionScores": "scores",
    "DecodeDetectionClasses": "labels"
}

for node in model.graph.node:
    for i, output in enumerate(node.output):
        if output in old_new_names:
            node.output[i] = old_new_names[output]

for output in model.graph.output:
    if output.name in old_new_names:
        output.name = old_new_names[output.name]

onnx.save(model, "yolov5s_ccpd.onnx")
