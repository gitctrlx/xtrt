# YOLO Model ONNX Modifier

> Please note that the scripts mentioned in the document below should be run within the **xtrt** directory. The scripts in this document originate from `xtrt\tools\modify_onnx`.

**Install Python Dependencies**: Run the following command to install the required Python packages.

```sh
pip install requirements.txt
```

> The following script utilizes the following four libraries: `onnx`, `onnx-graphsurgeon`, `onnxruntime`, `polygraphy`, and `numpy`.



## Add yoloLayer plugin

This feature is implemented in the script `xtrt\tools\modify_onnxadd_yoloLayer.py`, which adds a decode plugin to the model in the model-only format to accelerate the decoding process during TensorRT inference. For detailed information about the plugin, please refer to [`model_convert.md`](doc/model_convert.md).

### Usage

The script accepts two command-line arguments: `input_model_path` and `output_model_path`, specifying the paths for the input and output models, respectively. It provides a usage guide if the required arguments are not supplied.

```sh
python3 add_yoloLayer.py <input_model_path> <output_model_path>
```

**Examples:**

First, you need to download and export the pre-trained weights in ONNX format from Hugging Face, noting that it should be in the model-only format: https://huggingface.co/CtrlX/JetYOLO/tree/main/model-only

> The model-only format excludes the decode part of the model. For detailed information about this part, please refer to [`model_convert.md`](doc/model_convert.md).

```
python3 tools/add_yoloLayer.py weights/yolov5s.onnx weights/yolov5s_yoloLayer.onnx
```

### Detailed Guide

#### Adding Sigmoid Before YOLO Nodes

The script defines a function `add_sigmoid_before_yolo_nodes` which iterates over specified YOLO nodes, adding a Sigmoid activation function before each. This modification is aimed at normalizing the output, which can enhance the performance of the model during inference.

#### Modifying the YOLO ONNX Model

The main functionality is encapsulated within the `modify_yolo_onnx_model` function. This function performs the following steps:

1. **Loading the Model**: The input ONNX model is loaded.
2. **Graph Modification**: Utilizes ONNX-GraphSurgeon to manipulate the model graph.
3. **Sigmoid Activation Addition**: Calls `add_sigmoid_before_yolo_nodes` to insert Sigmoid nodes.
4. **YoloLayer Integration**: Adds a custom `YoloLayer_TRT` node for post-processing.
5. **Model Export**: The modified graph is exported as an ONNX model.

> **Note：**To effectively use this script, it's essential to modify the parameters in the `decode_attrs` dictionary according to your specific requirements. 
>
> ```py
> decode_attrs = {
>         "max_stride": np.array([32], dtype=np.int32),  
>         "num_classes": np.array([80], dtype=np.int32),  
>         "anchors": np.array([10, 13, 16, 30, 33, 23, 30, 61, 62, 45, 59, 119, 116, 90, 156, 198, 373, 326], dtype=np.float32),
>         "prenms_score_threshold": np.array([0.25], dtype=np.float32)  
>     }
> ```
>
> Here's a guide on how to adjust these parameters:
>
> - `max_stride`: This parameter defines the maximum stride used in the model. Modify this value based on the architecture and configuration of your YOLO model.
> - `num_classes`: Set the number of classes your model is trained to detect. Adjust this value accordingly if your model has a different number of classes.
> - `anchors`: Adjust the anchor values based on the anchor boxes used during training. Ensure that the anchor values match those used during the training of your YOLO model.
> - `prenms_score_threshold`: Set the score threshold for pre-NMS (Non-Maximum Suppression) filtering. This threshold determines which detections are considered before applying NMS. Adjust this threshold based on your model's performance and requirements.
>
> Ensure that these parameters are adjusted appropriately to achieve optimal performance and desired results for your specific YOLO model.



## Remove YOLO decode

This script originates from `xtrt\tools\remove_decode.py`. It is used to remove the decoding part from YOLO models in ONNX format. Of course, you can directly modify the code of YOLO models in the official YOLO repository to export models without the decoding part. Alternatively, you can use our provided code to directly modify the ONNX model and remove the decoding part.

### Usage

1. **Load the ONNX Model**: Specify the path to your ONNX model.
2. **Define Target Nodes**: List the names of the nodes you wish to modify.
3. **Execute Modifications**: Run the script to apply changes to the graph.
4. **Save the Modified Model**: The script outputs a new ONNX model with your modifications.

**Example：**

> To tailor the script to your needs, replace the placeholders in the code with your actual data. These include the path to your ONNX model, the names of the target nodes to modify, the new names for the output nodes, and the path for saving the modified model.

```py
# Load ONNX model
model_path = '../../weights/yolov5s.onnx'  # Path to your model

# Target node names
target_nodes_names = [
    '/baseModel/head_module/convs_pred.0/Conv',
    '/baseModel/head_module/convs_pred.1/Conv',
    '/baseModel/head_module/convs_pred.2/Conv'
]  # Specify the nodes you want to replace

# New output names
new_output_names = ['613', '614', '615']  # Set new names for the output nodes

# Save the modified model
onnx.save(gs.export_onnx(graph), 'modified_model.onnx')  # Define the path to save your modified model
```

### Detailed Guide

#### Functionality

The script comprises functions to:

- **Find Consumers**: Locate all nodes that use a given tensor name as input.
- **Find Dependent Nodes**: Recursively identify all nodes that depend on the target nodes.
- **Modify and Clean the Graph**: Change the output tensor names of target nodes and remove unnecessary dependent nodes.

#### Usage Steps

1. **Finding Consumers**: The `find_consumers` function searches the graph for nodes consuming a specified tensor, aiding in identifying connections within the graph.
2. **Identifying Dependent Nodes**: `find_dependent_nodes` uses a recursive approach to find all nodes that depend on the specified target nodes, ensuring thorough graph cleanup.
3. **Modifying the Graph**: After specifying target nodes and new output names, the script modifies these nodes' output tensor names and updates the graph's outputs accordingly.
4. **Cleaning Up**: The script removes any nodes that are dependent on the modified nodes but are not among the specified target nodes, streamlining the graph for efficiency.
5. **Saving the Model**: The modified graph is exported and saved as a new ONNX model file, ready for further use or deployment.



## Get ONNX node information

This script originates from `xtrt/tools/onnx_info.py`, and its functionality is aimed at exporting node information from an ONNX model to a text file. This feature is very useful for inspecting the structure of ONNX models, including node types, inputs, outputs, and attributes, which can assist in debugging and model analysis.

### Usage

> To tailor the script to your needs, replace the placeholders in the code with your actual data. 

```py
model_path = "modified_model.onnx" 
output_file_path = "model_node_info3.txt"
```

### Detailed Guide

#### Function Overview

- **Purpose**: To extract and save detailed information about each node in an ONNX model's graph to a text file.
- **Inputs**:
  - `model_path`: A string specifying the path to the ONNX model file.
  - `output_file_path`: A string specifying the path where the output text file containing node information will be saved.
- **Output**: No direct output, but the function writes to a file at `output_file_path` with detailed information on each node within the model's graph.

#### Usage Steps

1. **Load Model**: The function begins by loading the ONNX model from the specified `model_path` using the `onnx.load` method.
2. **Iterate Over Nodes**: It then iterates over each node in the model's graph.
3. **Write Node Information to File**: For each node, the function writes its name, type, inputs, outputs, and attributes to the specified output file. Each node's information is separated by newlines for clarity.



## Compress ONNX model using Polygraphy

Sometimes, TensorRT may fail to import ONNX models. In such cases, it is often helpful to clean up the ONNX model by removing redundant nodes and folding constants. The `surgeon sanitize` subtool of `Polygraphy` provides this capability using `ONNX-GraphSurgeon` and `ONNX-Runtime`.

### Usage

```sh
polygraphy surgeon sanitize "./weights/yolov5s.onnx" \
    --fold-constant \
    -o ./weights/yolo_pss.onnx \
    > ./weights/pss-result.log
```

### Command Explanation

- `polygraphy surgeon sanitize`: Invokes the Polygraphy Surgeon's sanitize command, which optimizes the ONNX model.
- `"./weights/yolov5s.onnx"`: Specifies the path to the input ONNX model file.
- `--fold-constant`: Enables constant folding. This optimization evaluates and simplifies operations in the model that involve constant values.
- `-o ./weights/yolo_pss.onnx`: Defines the output path for the optimized model. The optimized model will be saved as `yolo_pss.onnx` in the `./weights` directory.
- `> ./weights/pss-result.log`: Redirects the command's output (including any messages or errors) to a log file named `pss-result.log` in the `./weights` directory.

