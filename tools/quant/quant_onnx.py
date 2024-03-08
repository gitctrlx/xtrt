import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from ppq import *                                       
from ppq.api import *
from PIL import Image

from letterbox import Letterbox

# modify configuration below:
DATA_DIRECTORY        = '../../data/coco'
ONNX_IMPORT_FILE      = "../../weights/yolov5s_trt8.onnx"
ONNX_EXPORT_FILE      = "../../engine"
TARGET_PLATFORM       = TargetPlatform.TRT_INT8           # choose your target platform
EXPORT_PLATFORM       = TargetPlatform.ONNXRUNTIME        # choose your export platform , ONNXRUNTIME:QDQ
MODEL_TYPE            = NetworkFramework.ONNX             # or NetworkFramework.CAFFE
INPUT_LAYOUT          = 'chw'                             # input data layout, chw or hwc
NETWORK_INPUTSHAPE    = [1, 3, 640, 640]                  # input shape of your network
CALIBRATION_BATCHSIZE = 1                                 # batchsize of calibration dataset
EXECUTING_DEVICE      = 'cuda'                            # 'cuda' or 'cpu'.
REQUIRE_ANALYSE       = True
TRAINING_YOUR_NETWORK = True                              # Set to True to fine-tune the network

print('Preparing to quantify your network, check the following settings:')
print(f'    CALIBDAATA DIRECTORY : {DATA_DIRECTORY}')
print(f'    CALIBRATION BATCHSIZE: {CALIBRATION_BATCHSIZE}')
print(f'    TARGET PLATFORM      : {TARGET_PLATFORM.name}')
print(f'    EXPORT_PLATFORM      : {EXPORT_PLATFORM.name}')
print(f'    NETWORK INPUTSHAPE   : {NETWORK_INPUTSHAPE}')

graph = None
if MODEL_TYPE == NetworkFramework.ONNX:
    graph = load_onnx_graph(onnx_import_file = ONNX_IMPORT_FILE)
assert graph is not None, 'Graph Loading Error, Check your input again.'


QS = QuantizationSettingFactory.default_setting()
if TRAINING_YOUR_NETWORK:
    QS.lsq_optimization = True
    QS.lsq_optimization_setting.steps = 500
    QS.lsq_optimization_setting.collecting_device = 'cuda'
# You can send operators with poor quantization back to FP32, but of course, you need to first confirm that your hardware supports the execution of FP32. You can use layerwise error analysis to identify those operators with poor quantization
QS.dispatching_table.append(operation='', platform=TargetPlatform.FP32) 

# ------------------------------------------------------------------------------------------
# If the last layer of your model is a plugin operator, such as yolo, nms, etc., 
# The following code serves as an example of EfficientNMS_TRT. 
# These four shapes: [1, 1], [1, 100, 4], [1, 100], [1, 100] correspond to the three outputs of the model.
# Note that the output type here should also be consistent with the original onnx model.
# ------------------------------------------------------------------------------------------
def efficientnms_trt_forward(*args, **kwards):
    return torch.zeros([1, 1], dtype=torch.int32).cuda(), torch.zeros([1, 100, 4],dtype=torch.float32).cuda(), \
        torch.zeros([1, 100],dtype=torch.float32).cuda(), torch.zeros([1, 100], dtype=torch.int32).cuda() 
register_operation_handler(efficientnms_trt_forward, 'EfficientNMS_TRT', platform=TargetPlatform.FP32)

coco_val = []
trans = transform=transforms.Compose([
    # transforms.Resize((640, 640)),
    Letterbox(size=(640, 640)),
    transforms.ToTensor(),
])

MAX_IMAGES = 400
count = 0
for file in os.listdir(path=f"{DATA_DIRECTORY}/val2017"):
    if count >= MAX_IMAGES:
        break
    path = os.path.join(f"{DATA_DIRECTORY}/val2017", file)
    img = Image.open(path).convert('RGB')
    img = trans(img)
    coco_val.append(img) # img is 0-1
    count += 1

dataloader = DataLoader(coco_val, batch_size=CALIBRATION_BATCHSIZE, shuffle=False)


# Attempt to install the necessary compilation environment or proceed without CUDA KERNEL by removing the with statement.
with ENABLE_CUDA_KERNEL():
    print('Quantizing network, this may take some time depending on your configuration:')
    quantized = quantize_native_model(
        setting=QS,
        model=graph,
        calib_dataloader=dataloader,
        calib_steps=32,
        input_shape=NETWORK_INPUTSHAPE,
        inputs=None,
        collate_fn=lambda x: x.to(EXECUTING_DEVICE),  
        platform=TARGET_PLATFORM,
        device=EXECUTING_DEVICE,
        do_quantize=True)
    
    # Create an executor for running the quantized network similar to torch.Module.
    # Perform this before exporting the model.
    executor = TorchExecutor(graph=quantized, device=EXECUTING_DEVICE)
    # output = executor.forward(input)

    # Quantization error is evaluated using the inverse of the Signal-to-Noise Ratio (SNR),
    # with a quantization error of 0.1 indicating that quantization noise represents about 10% of the total signal energy.
    # Graphwise_error_analyse measures cumulative error, with the final layer often showing significant cumulative error from all preceding layers
    print('Calculating network quantization error (SNR), aiming for less than 0.1 in the last layer for acceptable accuracy:')
    reports = graphwise_error_analyse(
        graph=quantized, running_device=EXECUTING_DEVICE, steps=32,
        dataloader=dataloader, collate_fn=lambda x: x.to(EXECUTING_DEVICE))
    for op, snr in reports.items():
        if snr > 0.1: ppq_warning(f'Significant cumulative quantization error in layer {op}, consider optimization.')

    if REQUIRE_ANALYSE:
        print('Calculating layer-wise quantization error (SNR), aiming for less than 0.1 for each layer for acceptable accuracy:')
        layerwise_error_analyse(graph=quantized, running_device=EXECUTING_DEVICE,
                                interested_outputs=None,
                                dataloader=dataloader, collate_fn=lambda x: x.to(EXECUTING_DEVICE))

    # Export the quantized model using export_ppq_graph function, adjusting the format based on the target platform.
    print('Quantization complete, generating target files:')
    export_ppq_graph(
        graph=quantized, 
        platform=EXPORT_PLATFORM,
        graph_save_to = os.path.join(ONNX_EXPORT_FILE, 'quantized.onnx'),
        config_save_to = os.path.join(ONNX_EXPORT_FILE, 'quant_cfg.json'))