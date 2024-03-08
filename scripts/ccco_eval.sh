./build/eval \
    "../../engine/yolo.plan" \
    "../../data/coco/filelist.txt" \
    "../../data/coco/val2017/" \
    "results.json" \
    0 \
    true \
    2 \
    1 3 640 640

python3 coco_eval.py