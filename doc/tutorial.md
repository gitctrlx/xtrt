# XTRT

## Inference

#### 1. (Optional) Data Preparation

Data is used for calibration during quantization. We plan to use the [COCO val dataset](http://images.cocodataset.org/zips/val2017.zip) for model quantization calibration work. Place the downloaded val2017 dataset in the `xtrt/data/coco` directory.

```bash
xtrt\
 └── data
    └── coco
        ├── annotations
        └── val2017
```

#### 2. Model Preparation

Please read the [🔖 Model Zoo](#-model-zoo) section for downloading. If you want to quickly start with the examples below, you can skip this step, as the `xtrt/weights` folder in the cloned repository contains a `yolov5s` ONNX model with `EfficientNMS plugin`. 

> For a detailed analysis of different plugins and how to export, please see [`doc/model_convert.md`](model_convert.md).

#### 3. Building the Engine

Once the dataset is ready, the next step is to construct the engine. Below is an example for building a YOLOv5s TensorRT engine, with the corresponding code located in [`scripts/build_engine.sh`](https://github.com/gitctrlx/JetYOLO/blob/main/scripts/build_engine.sh):

```bash
./build/xtrt/build \
    "./xtrt/weights/yolov5s_trt8.onnx" \    # ONNX Model File Path
    "./xtrt/engine/yolo.plan" \             # TensorRT Engine Save Path
    "int8" \                                # Quantization Precision
    3 \                                     # TRT Optimization Level
    1 1 1 \                                 # Dynamic Shape Parameters
    3 3 3 \							 
    640 640 640 \					   
    640 640 640 \					   
    550 \                                   # Calibration Iterations
    "./xtrt/data/coco/val2017" \            # Calibration Dataset Path
    "./xtrt/data/coco/filelist.txt" \       # Calibration Image List
    "./xtrt/engine/int8Cache/int8.cache" \  # Calibration File Save Path
    true \                                  # Timing Cache Usage
    false \                                 # Ignore Timing Cache Mismatch
    "./xtrt/engine/timingCache/timing.cache"# Timing Cache Save Path
```

For a detailed analysis of the code's parameters, please see the [detailed documentation](doc).

**Verify the engine: Executing Inference（xtrt's inference demo）**

> **Note**: Run the demo to test if the engine was built successfully.

- demo-1: Inferencing a single image using the built YOLO TensorRT engine. The following code is located in [`scripts/demo_yolo_det_img.sh`](https://github.com/gitctrlx/JetYOLO/blob/main/scripts/demo_yolo_det_img.sh)：


```bash
./build/xtrt/yolo_det_img \
    "./xtrt/engine/yolo_trt8.plan" \ # TensorRT Engine Save Path
    "./xtrt/media/demo.jpg" \        # Input Image Path
    "./xtrt/output/output.jpg"\      # Output Image Path
    2 \                              # Pre-processing Pipeline
    1 3 640 640                      # Input Model Tensor Values
```

- demo-2: Inferencing a video using the built YOLO TensorRT engine. The following code is located in [`scripts/demo_yolo_det_video.sh`](https://github.com/gitctrlx/JetYOLO/blob/main/scripts/demo_yolo_det_video.sh)：


```bash
./build/xtrt/yolo_det \
    "./xtrt/engine/yolo_trt8.plan" \ # TensorRT Engine Save Path
    "./xtrt/media/c3.mp4" \          # Input Video Path 
    "./xtrt/output/output.mp4"\      # Output Video Path
    2 \	                             # Pre-processing Pipeline
    1 3 640 640	                     # Input Model Tensor Values
```

Then you can find the output results in the `xtrt/output` folder.

> **Note**: It is recommended to directly run the script or copy the code within the script for execution, rather than copying and running the code with comments included above:
>
> ```bash
> chmod 777 ./scripts/demo_yolo_det_img.sh # Grant execution permission to the script.
> ./scripts/demo_yolo_det_img.sh
> ```
>
> For a detailed analysis of the code's parameters, please see the [detailed documentation](doc).

## TensorRT Plugin



## Tools

### deploy



### eval



### quant



### modify onnx



### Trex