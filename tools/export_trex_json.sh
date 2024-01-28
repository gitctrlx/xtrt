trtexec \
    --loadEngine=./engine/yolo.plan \
    --dumpRefit \
    --dumpProfile \
    --profilingVerbosity=detailed \
    --dumpLayerInfo \
    --exportLayerInfo=./engine/layer.json \
    --exportProfile=./engine/profile.json

    