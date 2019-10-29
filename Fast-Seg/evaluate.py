import tensorflow as tf
import keras.backend as K

"""

Evaluate
1. Metric: mIOU
	Get the trained model
	Run inference
	Evalaute metrics
		-- CPU Inference time 
		-- Evaluate FLOPS / GFLOPS

"""


class Evaluation(object):
	def __init__(self, params):
		if params["model_type"] == 'keras_model':
			self.graph = params["model_graph_file_path"]
			self.weights = params["model_weights_file_path"]


	def get_model_params(model):
	    run_meta = tf.RunMetadata()

	    #Get FLOPS
		opts = tf.profiler.ProfileOptionBuilder.float_operation()    
		flops = tf.profiler.profile(graph=K.get_session().graph, run_meta=run_meta, 
									cmd='op', options=opts)

		#Get trainable model parameters
		opts = tf.profiler.ProfileOptionBuilder.trainable_variables_parameter()    
		trainable_params = tf.profiler.profile(graph=K.get_session().graph, run_meta=run_meta, 
									cmd='op', options=opts)


	    params={'total_parameters': trainable_params.total_parameters, 'flops': flops.total_float_ops}


