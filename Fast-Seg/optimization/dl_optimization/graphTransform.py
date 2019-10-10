import tensorflow as tf
from tensorflow.tools.graph_transforms import TransformGraph
from tensorflow.python import ops
import os


"""Demonstrates optimization of frozen Tensorflow GraphDef models using quantization and removing training dependent nodes. It uses python flavor of Tensorflow Graph Transforms [https://github.com/tensorflow/tensorflow/tree/master/tensorflow/tools/graph_transforms]

This is heavily inspired from the following Google blog post : [https://medium.com/google-cloud/optimizing-tensorflow-models-for-serving-959080e9ddbf]

For Efficient based models, only their weights can be quantized (fake tensorflow quantization) since some operators are not supported yet.

	Requirements 
    ------------
    Python 3.X
    tensorflow 1.14X+


    References
    ----------
    https://medium.com/google-cloud/optimizing-tensorflow-models-for-serving-959080e9ddbf
"""


class GraphTransform:
    """Load and setup frozen GraphDef file

		Args
		----
		model_params (dict)
		
	"""

    def __init__(self, model_params):
        """
		"""
        self.model_type = model_params["model_type"]
        self.model_dir = model_params["model_dir"]
        self.graph_filename = model_params["graph_filename"]
        self.output_node = model_params["output_node"]
        self.graph_def = self.get_graph_def_from_file(
            os.path.join(model_params["model_dir"], model_params["graph_filename"])
        )

    def describe_graph(self, graph_def, show_nodes=False):
        """Print nodes in the frozen tensorflow graph

			Args
			----
				graph_def(tf.GraphDef): Frozen Tensorflow GraphDef

			Returns
			------
				None

		"""
        print(
            "Input Feature Nodes: {}".format(
                [node.name for node in graph_def.node if node.op == "Placeholder"]
            )
        )
        print("")
        print(
            "Unused Nodes: {}".format(
                [node.name for node in graph_def.node if "unused" in node.name]
            )
        )
        print("")
        print(
            "Output Nodes: {}".format(
                [
                    node.name
                    for node in graph_def.node
                    if ("predictions" in node.name or "softmax" in node.name)
                ]
            )
        )
        print("")
        print(
            "Quantization Nodes: {}".format(
                [node.name for node in graph_def.node if "quant" in node.name]
            )
        )
        print("")
        print(
            "Constant Count: {}".format(
                len([node for node in graph_def.node if node.op == "Const"])
            )
        )
        print("")
        print(
            "Variable Count: {}".format(
                len([node for node in graph_def.node if "Variable" in node.op])
            )
        )
        print("")
        print(
            "Identity Count: {}".format(
                len([node for node in graph_def.node if node.op == "Identity"])
            )
        )
        print("", "Total nodes: {}".format(len(graph_def.node)), "")

        if show_nodes == True:
            for node in graph_def.node:
                print("Op:{} - Name: {}".format(node.op, node.name))

    def get_graph_def_from_file(self, graph_filepath):
        """Read frozen TF GraphDef file

			Args
			----
				graph_filepath(string)

			Returns
			-------
				tf.GraphDef object
		"""
        with ops.Graph().as_default():
            with tf.gfile.GFile(graph_filepath, "rb") as f:
                graph_def = tf.GraphDef()
                graph_def.ParseFromString(f.read())
                return graph_def

    def optimize_graph(self, transforms, optimized_file_name=None):
        """Applies graph transforms to the frozen GraphDef file & stores the optimized graph

			Args
			----
				transforms(list)

			Returns
			-------
				None

		"""
        if optimized_file_name is None:
            file_name, extension = os.path.splitext(graph_filename)
            self.optimized_file_name = file_name + "_optimized" + extension
        else:
            self.optimized_file_name = optimized_file_name

        input_names = []
        output_names = [self.output_node]
        print("Loading TF GraphDef file")
        graph_def = self.get_graph_def_from_file(
            os.path.join(self.model_dir, self.graph_filename)
        )
        print("===============Graph Loaded==================")
        print("===============Optimizing based on transforms : ")
        print(transforms)
        optimized_graph_def = TransformGraph(
            graph_def, input_names, output_names, transforms
        )
        tf.train.write_graph(
            optimized_graph_def,
            logdir=self.model_dir,
            as_text=False,
            name=optimized_file_name,
        )
        print("===============Graph Optimized==========")


if __name__ == "__main__":
    # specify the model configuration params; the input node and output node should have exact names
    base_dir = "../../data/assets/saved_models"
    model_params = {
        "model_type": "TF_GraphDef",
        "model_dir": base_dir,
        "graph_filename": "Unet_EB0_tf_model.pb",
        "output_node": "sigmoid/Sigmoid",
    }
    gt = GraphTransform(model_params)
    gt.describe_graph(gt.graph_def)

    # try different transforms
    # For complete list, see [https://github.com/tensorflow/tensorflow/tree/master/tensorflow/tools/graph_transforms]
    transforms = [
        'strip_unused_nodes(type=float, shape="1,128,128,3")',
        "remove_nodes(op=Identity)",
        "fold_constants(ignore_errors=true)",
        "fold_batch_norms",
        "quantize_weights",
    ]
    gt.optimize_graph(transforms, "chinmay_optimized.pb")
