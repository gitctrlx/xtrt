import onnx
import onnx_graphsurgeon as gs

def find_consumers(graph, tensor_name):
    """查找并返回所有使用给定张量名称作为输入的节点。"""
    consumers = []
    for node in graph.nodes:
        if tensor_name in [tensor.name for tensor in node.inputs]:
            consumers.append(node)
    return consumers

def find_dependent_nodes(graph, target_nodes):
    """递归找到所有依赖于目标节点的节点，使用节点名称作为标识。"""
    dependent_nodes_names = set()

    def recurse(node):
        consumers = find_consumers(graph, node.outputs[0].name)
        for consumer in consumers:
            if consumer.name not in dependent_nodes_names:
                dependent_nodes_names.add(consumer.name)
                recurse(consumer)

    for target_node in target_nodes:
        recurse(target_node)

    # 通过名称找回节点对象
    dependent_nodes = [node for node in graph.nodes if node.name in dependent_nodes_names]
    return dependent_nodes

# 加载ONNX模型
model_path = '../../weights/yolov5s2.onnx'
graph = gs.import_onnx(onnx.load(model_path))

# 目标节点名称
target_nodes_names = ['/baseModel/head_module/convs_pred.0/Conv', '/baseModel/head_module/convs_pred.1/Conv', '/baseModel/head_module/convs_pred.2/Conv']
target_nodes = [node for node in graph.nodes if node.name in target_nodes_names]

# 找到所有依赖的节点
dependent_nodes = find_dependent_nodes(graph, target_nodes)
dependent_node_names = {node.name for node in dependent_nodes}
target_node_names = {node.name for node in target_nodes}

# 新的输出名字列表
new_output_names = ['613', '614', '615']

# 确保有足够的新名字为每个目标节点指定一个名字
assert len(target_nodes) == len(new_output_names)

# 修改目标节点的输出张量名字
for node, new_name in zip(target_nodes, new_output_names):
    # 假设每个节点只有一个输出
    if len(node.outputs) > 0:
        node.outputs[0].name = new_name

# 更新图的输出为新的输出张量
graph.outputs = [node.outputs[0] for node in target_nodes]
nodes_to_remove_names = dependent_node_names - target_node_names
all_nodes_to_remove = [node for node in graph.nodes if node.name in nodes_to_remove_names]

# 从图中移除这些节点
graph.nodes = [node for node in graph.nodes if node not in all_nodes_to_remove]

# 更新图的输出
graph.outputs = [node.outputs[0] for node in target_nodes]

# 清理图，移除悬空的节点和张量
graph.cleanup()

# 保存修改后的模型
onnx.save(gs.export_onnx(graph), 'modified_model.onnx')
