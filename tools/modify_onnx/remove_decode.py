import onnx
import onnx_graphsurgeon as gs

def find_consumers(graph, tensor_name):
    """Find and return all nodes that use the given tensor name as input."""
    consumers = []
    for node in graph.nodes:
        if tensor_name in [tensor.name for tensor in node.inputs]:
            consumers.append(node)
    return consumers

def find_dependent_nodes(graph, target_nodes):
    """Recursively find all nodes that depend on the target node, using the node name as the identifier."""
    dependent_nodes_names = set()

    def recurse(node):
        consumers = find_consumers(graph, node.outputs[0].name)
        for consumer in consumers:
            if consumer.name not in dependent_nodes_names:
                dependent_nodes_names.add(consumer.name)
                recurse(consumer)

    for target_node in target_nodes:
        recurse(target_node)

    # Retrieve node objects by name
    dependent_nodes = [node for node in graph.nodes if node.name in dependent_nodes_names]
    return dependent_nodes

# Load ONNX model
model_path = '../../weights/yolov5s.onnx'
graph = gs.import_onnx(onnx.load(model_path))

# Target node name
target_nodes_names = ['/baseModel/head_module/convs_pred.0/Conv', '/baseModel/head_module/convs_pred.1/Conv', '/baseModel/head_module/convs_pred.2/Conv']
target_nodes = [node for node in graph.nodes if node.name in target_nodes_names]

# Find all dependent nodes
dependent_nodes = find_dependent_nodes(graph, target_nodes)
dependent_node_names = {node.name for node in dependent_nodes}
target_node_names = {node.name for node in target_nodes}

# New output name list
new_output_names = ['613', '614', '615']

# Ensure that there are enough new names to assign a name to each target node
assert len(target_nodes) == len(new_output_names)

# Modify the output tensor name of the target node
for node, new_name in zip(target_nodes, new_output_names):
    # Assuming that each node has only one output
    if len(node.outputs) > 0:
        node.outputs[0].name = new_name

# Update the output of the graph to a new output tensor
graph.outputs = [node.outputs[0] for node in target_nodes]
nodes_to_remove_names = dependent_node_names - target_node_names
all_nodes_to_remove = [node for node in graph.nodes if node.name in nodes_to_remove_names]

# Remove these nodes from the graph
graph.nodes = [node for node in graph.nodes if node not in all_nodes_to_remove]

# Update the output of the graph
graph.outputs = [node.outputs[0] for node in target_nodes]

# Clean up the graph, remove dangling nodes and tensors
graph.cleanup()

# Save the modified model
onnx.save(gs.export_onnx(graph), 'modified_model.onnx')
