#!/usr/bin/env python
import json
import cv2
import os
import tensorflow as tf
import time 
import numpy as np
import keras
import datetime
import argparse
from keras import backend as K
from keras.models import load_model
from keras.models import model_from_json
from copy import copy
from threading import Thread
from tqdm import tqdm


class FixedDropout(keras.layers.Dropout):
   def _get_noise_shape(self, inputs):
      if self.noise_shape is None:
         return self.noise_shape

      symbolic_shape = keras.backend.shape(inputs)
      noise_shape = [symbolic_shape[axis] if shape is None else shape
                    for axis, shape in enumerate(self.noise_shape)]
      return tuple(noise_shape)



class WebStream:
   def __init__(self, src=0):
      self.stream = cv2.VideoCapture(src)
      (self.grabbed, self.frame) = self.stream.read()
 
      # stop the thread
      self.stopped = False

   def start(self):
      Thread(target=self.update, args=()).start()
      return self
 
   def update(self):
      while True:
         if self.stopped:
            self.stream.release()
            return
 
         (self.grabbed, self.frame) = self.stream.read()
 
   def read(self):
      return self.frame
 
   def stop(self):
      self.stopped = True






if __name__=='__main__':
   #assumes wecam resolution is 1920x1080

   parser = argparse.ArgumentParser()

   parser.add_argument("-I", "--input-device", type=int, default=0, dest='input_device',
               help="Device number input")

   parser.add_argument("-F", "--full-screen", type=int, default=0, dest='full_screen',
               help="Full screen")

   parser.add_argument("-H", "--webcam-height", type=int, default=1080, dest='webcam_height',
               help="Webcam height resolution")

   parser.add_argument("-W", "--webcam-width", type=int, default=1920, dest='webcam_width',
               help="Webcam width resolution")


   args = vars(parser.parse_args())

   split_h, split_w = int(args["webcam_height"]) - 150, int(args["webcam_width"]//2)
   h,w,c = 128, 128, 3

   full_dir_path = os.path.abspath(os.path.dirname(__file__))

   #load the model from graph & setup the weights
   print("===============Loading Model==============")
   #base_model_path = '/home/chinmay/Code/workspace/saved_models/models/eng/EB0/128/'
   base_model_path = "/./data/assets/saved_models/"
   base_model_full_path = full_dir_path + base_model_path
   with open(base_model_full_path + 'Unet_EB0_128_graph.json','r') as f:
       model_json = json.load(f)

   model = model_from_json(model_json, custom_objects = {"swish": tf.nn.swish, "FixedDropout": FixedDropout})
   model.load_weights(base_model_full_path + 'Unet_EB0_128_weights.h5')
   print("===============Model loaded==============")
   print("===============Warming up the graph==============")
   r = np.random.rand(1,h,w,c)
   for i in tqdm(range(500)):
      pr_mask = model.predict(r)
   print("===============Graph warmed up===============")


   ws = WebStream(src=args["input_device"])
   cam = ws.start()

   total_time = []
   frames = 0
   myframe = None
   try:
      while True:

         # Image pre-processing
         # Shrink image to model inout of 128*128*3
         # Blur input image for better visual appeal
         frame = cam.read()
         myframe = copy(frame)
         t1 = time.time()
         orig = cv2.resize(myframe, (split_h, split_w))
         orig_normal = np.divide(orig, 255, dtype=np.float32)
         orig_blur = cv2.blur(orig_normal,(int(split_h/16), int(split_w/16)),0)
         #image = cv2.resize(orig, (h,w), interpolation = cv2.INTER_AREA)
         image = cv2.resize(orig, (h,w))
         image = image[..., ::-1] # switch BGR to RGB
         image = np.divide(image, 255, dtype=np.float32)
         image = image[np.newaxis, ...]

         #model prediction
         pr_mask = model.predict(image)
         mask = pr_mask[..., 0].squeeze()

         # Image postprocessing; remove background and apply background blur for better visually appealing stream
         mask_dst = cv2.resize(mask, (split_h, split_w))
         mask_dst = cv2.blur(mask_dst,(19, 19),0)
         new_image = np.multiply(orig_normal, mask_dst[:,:,None], dtype=np.float32)
         mask=np.dstack( ( mask_dst, mask_dst, mask_dst) )
         new_image = np.where(mask > 0.5, new_image, orig_blur)

         #display the frame
         color_and_mask = np.concatenate((orig_normal, new_image), axis=1)
         total_time.append(time.time() - t1)
         
         if args["full_screen"]:
            cv2.namedWindow("Segmentation", cv2.WND_PROP_FULLSCREEN)
            cv2.setWindowProperty("Segmentation",cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)

         cv2.imshow("Segmentation", color_and_mask)
         if cv2.waitKey(1) & 0xFF == ord('q'):
            cam.stop()
            cv2.destroyAllWindows()
            break

         frames +=1
   except Exception as e:
      print(e)
      cam.stop()
      cv2.destroyAllWindows()


   print("FPS {t}".format(t = ( frames /  sum(total_time)) ))
   print("Total Frames {f}".format(f=frames) )

