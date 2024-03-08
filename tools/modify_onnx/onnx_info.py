import onnx
from onnx import helper

def export_onnx_node_info_to_file(model_path, output_file_path):
    # 加载模型
    model = onnx.load(model_path)
    graph = model.graph

    # 打开文件准备写入节点信息
    with open(output_file_path, "w") as file:
        # 遍历并写入每个节点的详细信息到文件
        for i, node in enumerate(graph.node):
            file.write(f"Node {i}: {node.name}\n")
            file.write(f"  Type: {node.op_type}\n")
            file.write(f"  Inputs: {', '.join(node.input)}\n")
            file.write(f"  Outputs: {', '.join(node.output)}\n")
            for attr in node.attribute:
                file.write(f"  Attribute: {attr.name}\n")
            file.write("\n")

# 使用示例
model_path = "modified_model.onnx" # 替换为你的模型路径
output_file_path = "model_node_info3.txt" # 输出文件路径，你可以根据需要修改这个路径
export_onnx_node_info_to_file(model_path, output_file_path)
