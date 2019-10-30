import os
import json
import keras
import tensorflow as tf
import numpy as np
from keras import backend as K
from keras.models import load_model
from keras.models import model_from_json

"""

Evaluate the model architecture metrics

	Get the trained model
	Metrics
		-- Evaluate FLOPS
		-- Model total parameters

"""


class FixedDropout(keras.layers.Dropout):
    def _get_noise_shape(self, inputs):
        if self.noise_shape is None:
            return self.noise_shape

        symbolic_shape = keras.backend.shape(inputs)
        noise_shape = [
            symbolic_shape[axis] if shape is None else shape
            for axis, shape in enumerate(self.noise_shape)
        ]
        return tuple(noise_shape)


class Evaluation(object):
    """Class to evaluate model architecture parameters
	"""

    def __init__(self, params):
        """Initialize model graph and weights

			Args:
			-----
				params(dict)

			Returns:
			-------
				None
		"""
        if params["model_type"] == "keras_model":
            self.graph = params["model_graph_file_path"]
            self.weights = params["model_weights_file_path"]

    def load_model(self):
        """Load model from graph

			Args:
			-----
				None

			Returns:
			--------
				None
		"""
        print("==================LOADING MODEL==================")
        # load the model from graph & setup the weights
        with open(self.graph, "r") as f:
            model_json = json.load(f)

        model = model_from_json(
            model_json,
            custom_objects={"swish": tf.nn.swish, "FixedDropout": FixedDropout},
        )
        model.load_weights(self.weights)
        print("==================MODEL LOADED==================")
        return model

    def get_model_params(self, input_shape):
        """Get model architecture metrics:
			Metrics: FLOPS and model parameters
			
			Args:
			-----
				None

			Returns:
			--------
				params(dict): model parameters
		"""

		
        r = np.random.rand(input_shape[0], input_shape[1], input_shape[2], input_shape[3]).astype(np.float32)
        input_data = tf.placeholder(tf.float32, input_shape, name = 'input_data')
        model = self.load_model()
        output = model(input_data)
        run_metadata = tf.RunMetadata()
        #CAUTION: Create a TF Session and run a sample input. Otherwise profiler gives incorrect FLOPS
        with tf.Session(graph=K.get_session().graph) as session:
            session.run(tf.global_variables_initializer())
            #output_val = session.run(output, {input_img: r})

            print(session.run(output, {input_data: r},
                       options=tf.RunOptions(
                           trace_level=tf.RunOptions.FULL_TRACE),
                       run_metadata=run_metadata))
           
            opts = tf.profiler.ProfileOptionBuilder.float_operation()
            flops = tf.profiler.profile( graph= tf.get_default_graph(),
              run_meta=run_metadata, cmd="op", options=opts
            )

            opts = tf.profiler.ProfileOptionBuilder.trainable_variables_parameter()
            trainable_params = tf.profiler.profile( graph= tf.get_default_graph(),
                run_meta=run_metadata, cmd="op", options=opts
            )

        params = {
            "total_parameters": trainable_params.total_parameters,
            "flops": flops.total_float_ops,
        }
        return params


if __name__ == "__main__":
    full_dir_path = os.path.abspath(os.path.dirname(__file__))
    base_dir = full_dir_path + "/data/assets/saved_models/"
    weights = base_dir + "Unet_EB0_128_weights.h5"
    graph = base_dir + "Unet_EB0_128_graph.json"

    model_params = {
        "model_type": "keras_model",
        "model_graph_file_path": graph,
        "model_weights_file_path": weights,
    }

    eval = Evaluation(model_params)
    input_data = (1,128,128,3)
    print(eval.get_model_params(input_data))
