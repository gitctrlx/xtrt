
./build/build \
    "./weights/yolov5s_trt8.onnx" \
    "./engine/yolo.plan" \
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
