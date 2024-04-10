import onnx
from onnx import helper

def export_onnx_node_info_to_file(model_path, output_file_path):
    model = onnx.load(model_path)
    graph = model.graph

    with open(output_file_path, "w") as file:
        for i, node in enumerate(graph.node):
            file.write(f"Node {i}: {node.name}\n")
            file.write(f"  Type: {node.op_type}\n")
            file.write(f"  Inputs: {', '.join(node.input)}\n")
            file.write(f"  Outputs: {', '.join(node.output)}\n")
            for attr in node.attribute:
                file.write(f"  Attribute: {attr.name}\n")
            file.write("\n")

model_path = "modified_model.onnx" 
output_file_path = "model_node_info3.txt"
export_onnx_node_info_to_file(model_path, output_file_path)
