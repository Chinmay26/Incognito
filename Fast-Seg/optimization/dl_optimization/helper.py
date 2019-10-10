import json
import tensorflow as tf
import os
import numpy as np
import keras
from keras import backend as K
from keras.models import load_model
from keras.models import model_from_json


class FixedDropout(keras.layers.Dropout):
    """FixedDropout Layer defined in the Efficient Net EB0-7 models. This is needed during graph conversion from Keras models to Tensorflow GraphDef

		Args
		----
			dropout(keras.layers.Dropout)

	"""

    def _get_noise_shape(self, inputs):
        """
			Args
			----
				inputs(numpy array)

			Returns
			-------
				tuple
		"""
        if self.noise_shape is None:
            return self.noise_shape

        symbolic_shape = keras.backend.shape(inputs)
        noise_shape = [
            symbolic_shape[axis] if shape is None else shape
            for axis, shape in enumerate(self.noise_shape)
        ]
        return tuple(noise_shape)


class GraphConvert(object):
    """Converts Keras model into Tensorflow GraphDef format saved in protobuf format

		Args
		----
			graph_file(str): file path of the frozen keras graph
			weights_file(str): weights file path of the keras model

	"""

    def __init__(self, graph_file, weights_file):
        """Initialize keras graph file and weights path

		"""
        self.graph_file = graph_file
        self.weights_file = weights_file

    def freeze_session(
        self, session, keep_var_names=None, output_names=None, clear_devices=True
    ):
        """Freezes the state of a session into a pruned computation graph.

		Creates a new computation graph where variable nodes are replaced by
		constants taking their current value in the session. The new graph will be
		pruned so subgraphs that are not necessary to compute the requested
		outputs are removed.
		@param session The TensorFlow session to be frozen.
		@param keep_var_names A list of variable names that should not be frozen,
		                      or None to freeze all the variables in the graph.
		@param output_names Names of the relevant graph outputs.
		@param clear_devices Remove the device directives from the graph for better portability.
		@return The frozen graph definition.

		References
		----------
			This function is copied from [https://stackoverflow.com/questions/45466020/how-to-export-keras-h5-to-tensorflow-pb]

		"""
        graph = session.graph
        with graph.as_default():
            freeze_var_names = list(
                set(v.op.name for v in tf.global_variables()).difference(
                    keep_var_names or []
                )
            )
            output_names = output_names or []
            output_names += [v.op.name for v in tf.global_variables()]
            input_graph_def = graph.as_graph_def()
            if clear_devices:
                for node in input_graph_def.node:
                    node.device = ""
            frozen_graph = tf.graph_util.convert_variables_to_constants(
                session, input_graph_def, output_names, freeze_var_names
            )
            return frozen_graph

    def load_keras_model(self, custom_objects=None):
        """Load Keras model from its frozen graph and weights file

			Args
			----
				custom_objects(dict): dictionary of custom model parts and their definitions

			Returns
			-------
				model(keras.model): loaded keras model

		"""
        with open(self.graph_file, "r") as f:
            model_json = json.load(f)

        model = model_from_json(model_json, custom_objects=custom_objects)
        model.load_weights(self.weights_file)
        return model

    def convert(self, tf_graph_file_name, custom_objects=None):
        """Convert keras models to TF models

			Args
			----
				tf_model_graph_name(str): name of the converted TF graph
				custom_objects(dict): custom parts of the model and their Tensorflow definitions

		"""
        K.clear_session()  # for multiple runs, clear the session
        keras_model = self.load_keras_model(custom_objects)
        tf_frozen_graph = self.freeze_session(
            K.get_session(), output_names=[out.op.name for out in keras_model.outputs]
        )
        tf.train.write_graph(
            tf_frozen_graph, base_dir, tf_graph_file_name, as_text=False
        )


if __name__ == "__main__":
    base_dir = "../../data/assets/saved_models"
    keras_graph_file = os.path.join(base_dir, "Unet_EB0_graph.json")
    keras_weight_file = os.path.join(base_dir, "Unet_EB0_weights.h5")
    print("=========Loading Keras model====================")
    gf = GraphConvert(keras_graph_file, keras_weight_file)
    print("=========Keras model loaded=====================")
    tf_graph_file_name = "Unet_EB0_tf_model.pb"
    # For Efficient Net based models, swish activation and FixedDropout need to be defined
    custom_objects = {"swish": tf.nn.swish, "FixedDropout": FixedDropout}
    print("========Converting keras model to TF graph======")
    try:
        gf.convert(tf_graph_file_name, custom_objects)
    except Exception as err:
        print("=========Faield to convert TF model======")
        print(err)
    print("========Conversion complete=====================")
