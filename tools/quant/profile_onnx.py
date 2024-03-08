import torch
import torch.profiler
from ppq import *
from ppq.api import *
from ppq.core import TargetPlatform
from tqdm import tqdm
from ppq.api import register_operation_handler

def happyforward(*args, **kwards):
    return \
        torch.zeros([1, 1], dtype=torch.int32).cuda(), \
        torch.zeros([1, 100, 4],dtype=torch.float32).cuda(), \
        torch.zeros([1, 100],dtype=torch.float32).cuda(), \
        torch.zeros([1, 100], dtype=torch.int32).cuda() 
register_operation_handler(happyforward, 'EfficientNMS_TRT', platform=TargetPlatform.FP32)


sample_input = [torch.rand(1, 3, 640, 640) for i in range(32)]
ir = quantize_onnx_model(
    onnx_import_file='yolov5s_trt8.onnx',
    calib_dataloader=sample_input,
    calib_steps=16,
    do_quantize=False,  # 根据需要调整，如果要进行量化，设置为 True
    input_shape=None,
    collate_fn=lambda batch: torch.stack(batch).to('cuda'),
    inputs=torch.rand(1, 3, 640, 640).to('cuda'),
    platform=TargetPlatform.TRT_INT8
)
executor = TorchExecutor(ir)

with torch.profiler.profile(
    schedule=torch.profiler.schedule(wait=2, warmup=2, active=6, repeat=1),
    on_trace_ready=torch.profiler.tensorboard_trace_handler(dir_name='log/'),
    activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
    with_stack=True,
) as profiler:
    with torch.no_grad():
        for batch_idx in tqdm(range(16), desc='Profiling ...'):
            executor.forward(sample_input[batch_idx].to('cuda'))
            profiler.step()
