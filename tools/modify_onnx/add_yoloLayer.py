import onnx_graphsurgeon as onnx_gs
import numpy as np
import onnx
import sys

def add_sigmoid_before_yolo_nodes(yolo_graph, nodes):
    sigmoid_nodes = []
    for node in nodes:
        sigmoid_out = onnx_gs.Variable(f"{node.name}_sigmoid", dtype=np.float32)
        sigmoid_node = onnx_gs.Node(
            op="Sigmoid",
            name=f"{node.name}_sigmoid",
            inputs=[node],
            outputs=[sigmoid_out]
        )
        sigmoid_nodes.append(sigmoid_node)

    yolo_graph.nodes.extend(sigmoid_nodes)
    return [sigmoid_node.outputs[0] for sigmoid_node in sigmoid_nodes]

def modify_yolo_onnx_model(input_model_path, output_model_path):
    model_onnx = onnx.load(input_model_path)
    yolo_graph = onnx_gs.import_onnx(model_onnx)

    p3, p4, p5 = yolo_graph.outputs[:3]

    sigmoid_p3, sigmoid_p4, sigmoid_p5 = add_sigmoid_before_yolo_nodes(yolo_graph, [p3, p4, p5])

    decode_out_0 = onnx_gs.Variable("num_dets", dtype=np.int32)
    decode_out_1 = onnx_gs.Variable("boxes", dtype=np.float32)
    decode_out_2 = onnx_gs.Variable("scores", dtype=np.float32)
    decode_out_3 = onnx_gs.Variable("labels", dtype=np.int32)

    decode_attrs = {
        "max_stride": np.array([32], dtype=np.int32),  
        "num_classes": np.array([80], dtype=np.int32),  
        "anchors": np.array([10, 13, 16, 30, 33, 23, 30, 61, 62, 45, 59, 119, 116, 90, 156, 198, 373, 326], dtype=np.float32),
        "prenms_score_threshold": np.array([0.25], dtype=np.float32)  
    }

    decode_plugin = onnx_gs.Node(
        op="YoloLayer_TRT",
        name="YoloLayer",
        inputs=[sigmoid_p3, sigmoid_p4, sigmoid_p5],
        outputs=[decode_out_0, decode_out_1, decode_out_2, decode_out_3],
        attrs=decode_attrs
    )
    yolo_graph.nodes.append(decode_plugin)
    yolo_graph.outputs = decode_plugin.outputs
    yolo_graph.cleanup().toposort()
    modified_model = onnx_gs.export_onnx(yolo_graph)

    onnx.save(modified_model, output_model_path)
    print(f"[I] Modified model saved to {output_model_path}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python modify_onnx.py <input_model_path> <output_model_path>")
        sys.exit(1)

    input_model_path = sys.argv[1]
    output_model_path = sys.argv[2]

    modify_yolo_onnx_model(input_model_path, output_model_path)
    
# usage
# python3 tools/modify_onnx.py weights/yolov5s.onnx weights/yolov5s_yoloLayer.onnx
# python3 tools/modify_onnx.py weights/yolov6_n_yl.onnx weights/yolov6n_yoloLayer.onnx
