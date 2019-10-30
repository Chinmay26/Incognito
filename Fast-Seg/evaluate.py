import os
import json
import tensorflow as tf
import keras
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
        noise_shape = [symbolic_shape[axis] if shape is None else shape
                       for axis, shape in enumerate(self.noise_shape)]
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
		if params["model_type"] == 'keras_model':
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
		#load the model from graph & setup the weights
		with open(self.graph,'r') as f:
		    model_json = json.load(f)

		model = model_from_json(model_json, custom_objects = {"swish": tf.nn.swish, "FixedDropout": FixedDropout})
		model.load_weights(self.weights)
		print("==================MODEL LOADED==================")
		return model


	def get_model_params(self):
		"""Get model architecture metrics:
			Metrics: FLOPS and model parameters

			Args:
			-----
				None

			Returns:
			--------
				params(dict): model parameters
		"""
		model = self.load_model()
		run_meta = tf.RunMetadata()

		#Get FLOPS
		opts = tf.profiler.ProfileOptionBuilder.float_operation()    
		flops = tf.profiler.profile(graph=K.get_session().graph, run_meta=run_meta, 
									cmd='op', options=opts)

		#Get total model parameters
		opts = tf.profiler.ProfileOptionBuilder.trainable_variables_parameter()    
		trainable_params = tf.profiler.profile(graph=K.get_session().graph, run_meta=run_meta, 
									cmd='op', options=opts)

		params={'total_parameters': trainable_params.total_parameters, 'flops': flops.total_float_ops}
		return params


if __name__=='__main__':
	full_dir_path = os.path.abspath(os.path.dirname(__file__))
	base_dir = full_dir_path +  "/data/assets/saved_models/"
	weights = base_dir + 'Unet_EB0_128_weights.h5'
	graph = base_dir + 'Unet_EB0_128_graph.json'
	model_params = {
		"model_type": "keras_model",
		"model_graph_file_path": graph,
		"model_weights_file_path": weights
	}

	eval = Evaluation(model_params)
	print(eval.get_model_params())


