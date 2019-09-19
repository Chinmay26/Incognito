

'''

This class has been sourced from : https://colab.research.google.com/github/tensorflow/models/blob/master/research/deeplab/deeplab_demo.ipynb
and modified as per custom requirements

'''


import os
from io import BytesIO
import tarfile
import tempfile
from six.moves import urllib

from matplotlib import gridspec
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image

import tensorflow as tf


class Model:
	"""Class to run inference on pretrained tensorflow models """
	INPUT_TENSOR_NAME = 'ImageTensor:0'
	OUTPUT_TENSOR_NAME = 'SemanticPredictions:0'
	INPUT_SIZE = 513
	FROZEN_GRAPH_NAME = 'frozen_inference_graph'

	def __init__(self, graph_def_pb_file):
		""" Initialize TF graph with pretrained TF model 

			Args:

			Returns:

		"""
		self.graph = tf.Graph()
		graph_def = None

		with tf.io.gfile.GFile(graph_def_pb_file, "rb") as f:
			graph_def = tf.compat.v1.GraphDef()
			graph_def.ParseFromString(f.read())


		if graph_def is None:
			raise RuntimeError('Cannot locate trained model weights')

		with self.graph.as_default():
			tf.import_graph_def(graph_def, name='')

		self.sess = tf.compat.v1.Session(graph=self.graph)

	def resize_image(self, image):
		""" Resize image based on model input node  

			Args: single input image

			Returns: resized image according to model graph input node

		"""
		w, h = image.size
		resize_ratio = 1.0 * self.INPUT_SIZE / max(w, h)
		target_size = (int(resize_ratio * w), int(resize_ratio * h))
		resized_image = image.convert('RGB').resize(target_size, Image.ANTIALIAS)
		return resized_image

	def run(self, image):
		""" Runs inference on an image
			
			Args: single input image

			Returns: 
				resized_image: input image resized from original image
				seg_map: segmentation map returned by the model

		"""
		w,h = image.size
		resized_image = self.resize_image(image)
		batch_seg_map = self.sess.run(
			self.OUTPUT_TENSOR_NAME,
			feed_dict={self.INPUT_TENSOR_NAME: [np.asarray(resized_image)]})
		seg_map = batch_seg_map[0]
		return resized_image, seg_map


	def create_pascal_label_colormap(self):
		"""Creates a label colormap used in PASCAL VOC segmentation benchmark.

		Returns:
		A Colormap for visualizing segmentation results.
		"""
		colormap = np.zeros((256, 3), dtype=int)
		ind = np.arange(256, dtype=int)

		for shift in reversed(range(8)):
			for channel in range(3):
			  colormap[:, channel] |= ((ind >> channel) & 1) << shift
			ind >>= 3

		return colormap


	def label_to_color_image(self, label):
		"""Adds color defined by the dataset colormap to the label.

		Args:
		label: A 2D array with integer type, storing the segmentation label.

		Returns:
		result: A 2D array with floating type. The element of the array
		  is the color indexed by the corresponding element in the input label
		  to the PASCAL color map.

		Raises:
		ValueError: If label is not of rank 2 or its value is larger than color
		  map maximum entry.
		"""
		if label.ndim != 2:
			raise ValueError('Expect 2-D input label')

		colormap = create_pascal_label_colormap()

		if np.max(label) >= len(colormap):
			raise ValueError('label value too large.')

		return colormap[label]


	def vis_segmentation(self, image, seg_map):
		"""Visualizes input image, segmentation map and overlay view."""
		plt.figure(figsize=(15, 5))
		grid_spec = gridspec.GridSpec(1, 4, width_ratios=[6, 6, 6, 1])

		plt.subplot(grid_spec[0])
		plt.imshow(image)
		plt.axis('off')
		plt.title('input image')

		plt.subplot(grid_spec[1])
		seg_image = label_to_color_image(seg_map).astype(np.uint8)
		plt.imshow(seg_image)
		plt.axis('off')
		plt.title('segmentation map')

		plt.subplot(grid_spec[2])
		plt.imshow(image)
		plt.imshow(seg_image, alpha=0.7)
		plt.axis('off')
		plt.title('segmentation overlay')


if __name__ == '__main__':
	model_path = '/home/chinmay/Code/workspace/github/deeplab_models/deeplabv3_mnv2_dm05_pascal_trainaug/frozen_inference_graph.pb'
	image_path = '/home/chinmay/Pictures/image1.jpg'
	model = Model(model_path)
	original_im = Image.open(image_path)
	resized_im, seg_map = model.run(original_im)

