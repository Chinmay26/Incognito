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