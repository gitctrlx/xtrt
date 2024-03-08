./build/build \
    "./weights/yolov5s_trt8.onnx" \
    "./engine/yolo_trt8.plan" \
    "fp16" \
    3 \
    1 1 1 \
    3 3 3 \
    640 640 640 \
    640 640 640 \
    200 \
    "./calibdata/" \
    "./calibdata/filelist.txt" \
    "./engine/int8Cache/int8.cache" \
    true \
    false \
    "./engine/timingCache/timing.cache"

./build/build \
    "./weights/yolov5s_yoloLayer.onnx" \
    "./engine/yolo.plan" \
    "fp32" \
    3 \
    1 1 1 \
    3 3 3 \
    640 640 640 \
    640 640 640 \
    200 \
    "./calibdata/" \
    "./calibdata/filelist.txt" \
    "./engine/int8Cache/int8.cache" \
    true \
    false \
    "./engine/timingCache/timing.cache"

./build/build \
    "./weights/quantized_notrainqdq.onnx" \
    "./engine/yolo.plan" \
    "int8" \
    3 \
    1 1 1 \
    3 3 3 \
    640 640 640 \
    640 640 640 \
    550 \
    "./data/coco/val2017" \
    "./data/coco/filelist.txt" \
    "./engine/int8Cache/int8.cache" \
    true \
    false \
    "./engine/timingCache/timing.cache"