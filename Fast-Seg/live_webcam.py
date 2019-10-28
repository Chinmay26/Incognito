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

"""
Demonstrates live webcam background segmentation on average CPU workstations.


"""


class FixedDropout(keras.layers.Dropout):
    """
      Custom dropout layer defined in EfficientNet; the definition is needed while loading the graph
   """

    def _get_noise_shape(self, inputs):
        """Noise shape for input

         Args:
         ----
            inputs(np.array)

         Returns:
         -------
            tuple(noise): noise based on input dimesnions
      """
        if self.noise_shape is None:
            return self.noise_shape

        symbolic_shape = keras.backend.shape(inputs)
        noise_shape = [
            symbolic_shape[axis] if shape is None else shape
            for axis, shape in enumerate(self.noise_shape)
        ]
        return tuple(noise_shape)


class WebStream:
    """Class to acquire webcam frames in a threaded fashion
   """

    def __init__(self, src=0):
        """Initialize the webcam stream

         Args:
         -----
            src(int or video_file_path): source to capture frames

         Returns:
         -------
            None
      """
        self.stream = cv2.VideoCapture(src)
        (self.grabbed, self.frame) = self.stream.read()
        self.stopped = False

    def start(self):
        """Start separate thread to acquire frames

         Args:
         -----
            None

         Returns:
         --------
            None
      """
        Thread(target=self.update, args=()).start()
        return self

    def update(self):
        """Update the frame which is captured

         Args:
         -----
            None

         Returns:
         --------
            None
      """
        while True:
            if self.stopped:
                self.stream.release()
                return

            (self.grabbed, self.frame) = self.stream.read()

    def read(self):
        """Update the frame which is currently read

         Args:
         -----
            None

         Returns:
         --------
            frame(np.array): input image frame
      """
        return self.frame

    def stop(self):
        """Stop capturing frames from the input stream

         Args:
         -----
            None

         Returns:
         --------
            None
      """
        self.stopped = True


class Segmentation(object):
    def __init__(self, args):
        self.split_h, self.split_w = (
            int(args["webcam_height"]) - 150,
            int(args["webcam_width"] // 2),
        )
        self.h, self.w, self.c = 128, 128, 3
        self.threshold = 0.5
        self.base_model_path = args["base_model_path"]

    def load_model(self):
        """Load keras model based on path
               Args:
               -----
                  base_model_path(str): path to the frozen model

               Returns:
               --------
                  model(keras.model): model loaded with weights
      """
        full_dir_path = os.path.abspath(os.path.dirname(__file__))

        # load the model from graph & setup the weights
        print("===============Loading Model==============")
        base_model_full_path = full_dir_path + self.base_model_path
        with open(base_model_full_path + "Unet_EB0_128_graph.json", "r") as f:
            model_json = json.load(f)

        model = model_from_json(
            model_json,
            custom_objects={"swish": tf.nn.swish, "FixedDropout": FixedDropout},
        )
        model.load_weights(base_model_full_path + "Unet_EB0_128_weights.h5")
        print("===============Model loaded==============")
        return model

    def preprocess(self, frame):
        """Preprocess input image
               Args:
               -----
                  frame(np.array): input image

               Returns:
               --------
                  images(tuple): tuple of preprocessed images
      """
        orig = cv2.resize(frame, (self.split_h, self.split_w))
        orig_normal = np.divide(orig, 255, dtype=np.float32)
        orig_blur = cv2.blur(
            orig_normal, (int(self.split_h / 16), int(self.split_w / 16)), 0
        )
        image = cv2.resize(orig, (self.h, self.w), interpolation=cv2.INTER_AREA)
        image = image[..., ::-1]  # switch BGR to RGB
        image = np.divide(image, 255, dtype=np.float32)
        image = image[np.newaxis, ...]
        return image, orig_normal, orig_blur

    def postprocess(self, mask, orig_normal, orig_blur):
        """Preprocess input image
               Args:
               -----
                  mask(np.array): input masked image
                  orig_normal(np.array): input normalized image
                  orig_blur(np.array): input blurred image

               Returns:
               --------
                  new_image(np.array): final background segmented masked image
      """
        # remove background and apply background blur for better visually appealing stream
        mask_dst = cv2.resize(
            mask, (self.split_h, self.split_w), interpolation=cv2.INTER_CUBIC
        )
        mask_dst = cv2.blur(mask_dst, (15, 15), 0)
        new_image = np.multiply(orig_normal, mask_dst[:, :, None], dtype=np.float32)
        mask = np.dstack((mask_dst, mask_dst, mask_dst))
        new_image = np.where(mask > self.threshold, new_image, orig_blur)
        return new_image


if __name__ == "__main__":
    # assumes wecam resolution is 1920x1080

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-I",
        "--input-device",
        type=int,
        default=0,
        dest="input_device",
        help="Device number input",
    )

    parser.add_argument(
        "-F",
        "--full-screen",
        type=int,
        default=0,
        dest="full_screen",
        help="Full screen",
    )

    parser.add_argument(
        "-H",
        "--webcam-height",
        type=int,
        default=1080,
        dest="webcam_height",
        help="Webcam height resolution",
    )

    parser.add_argument(
        "-W",
        "--webcam-width",
        type=int,
        default=1920,
        dest="webcam_width",
        help="Webcam width resolution",
    )

    parser.add_argument(
        "-P",
        "--model_path",
        type=str,
        default="/./data/assets/saved_models/",
        dest="base_model_path",
        help="Path to frozen model",
    )

    args = vars(parser.parse_args())
    h, w, c = 128, 128, 3

    seg = Segmentation(args)
    model = seg.load_model()

    print("===============Warming up the graph==============")
    r = np.random.rand(1, h, w, c)
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
            frame = cam.read()
            myframe = copy(frame)
            t1 = time.time()

            # preprocess
            image, orig_normal, orig_blur = seg.preprocess(myframe)

            # model prediction
            pr_mask = model.predict(image)
            mask = pr_mask[..., 0].squeeze()

            # postprocess
            new_image = seg.postprocess(mask, orig_normal, orig_blur)

            # display the frame
            color_and_mask = np.concatenate((orig_normal, new_image), axis=1)
            total_time.append(time.time() - t1)

            if args["full_screen"]:
                cv2.namedWindow("Segmentation", cv2.WND_PROP_FULLSCREEN)
                cv2.setWindowProperty(
                    "Segmentation", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN
                )

            cv2.imshow("Segmentation", color_and_mask)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                cam.stop()
                cv2.destroyAllWindows()
                break

            frames += 1
    except Exception as e:
        print(e)
        cam.stop()
        cv2.destroyAllWindows()

    print("FPS {t}".format(t=(frames / sum(total_time))))
    print("Total Frames {f}".format(f=frames))
