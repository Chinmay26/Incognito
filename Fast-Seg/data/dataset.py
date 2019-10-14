"""
Dataset handler to load images and masks necessary for training

This is inspired by https://github.com/qubvel/segmentation_models/blob/master/examples/binary%20segmentation%20(camvid).ipynb & modified to suit my requirements

"""

import os
import numpy as np
import keras
import json
import cv2
from scipy.io import loadmat


class Dataset(object):
    """
        Dataset handler to laod images and segmentation masks
    """

    def __init__(
        self,
        images_dir,
        masks_dir,
        test=False,
        total_classes=["person"],
        img_size=(128, 128),
    ):
        """Initialize the dataset

            Args
            ----
                images_dir(str): path to the images dir
                masks_dir(str): path to the masks dir
                test(bool): Load dataset in training or test mode
                total_classes(list): total segmentation classes
                img_size(tuple): image resoltion

            Returns
            ------
                None

        """
        self.img_resize = img_size
        self.ids = os.listdir(images_dir)
        self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids]
        self.masks_fps = []

        for image_id in self.ids:
            img_name, extension = os.path.splitext(image_id)
            mask_name = img_name + ".png"
            self.masks_fps.append(os.path.join(masks_dir, mask_name))

        # convert str names to class values on masks
        self.class_values = [total_classes.index(cls.lower()) for cls in total_classes]

    def __getitem__(self, i):
        """Get specific items from the dataset

            Args
            ----
                i (int): Get ith items

            Returns
                image,mask(tuple): Get tuple of image and segmentation mask
        """
        image = cv2.imread(self.images_fps[i]).astype(np.float32)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, self.img_resize)
        image = np.divide(image, 255.0, dtype=np.float32)

        mask = cv2.imread(self.masks_fps[i], cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, self.img_resize).astype(np.uint8)
        mask = np.expand_dims(mask, axis=2)

        return image, mask

    def __len__(self):
        return len(self.ids)


class Dataloder(keras.utils.Sequence):
    """.
        Data Generator to load images & masks in batches
    """

    def __init__(self, dataset, batch_size=1, shuffle=False):
        """Initialize the data generator

            Args
            ----
                dataset(Dataset): the dataset to be batch loaded
                batch_size(int): size of the batch to be generated
                shuffle(bool): shuffle the batch before being sent
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indexes = np.arange(len(dataset))

        self.on_epoch_end()

    def __getitem__(self, i):
        """Get specific items from the dataset

            Args
            ----
                i (int): Get ith items

            Returns
                batch(list): Get list of tuple of image and segmentation masks
        """
        # collect batch data
        start = i * self.batch_size
        stop = (i + 1) * self.batch_size
        data = []
        for j in range(start, stop):
            data.append(self.dataset[j])

        # transpose list of lists
        batch = [np.stack(samples, axis=0) for samples in zip(*data)]

        return batch

    def __len__(self):
        """Denotes the number of batches per epoch"""
        return len(self.indexes) // self.batch_size

    def on_epoch_end(self):
        """Callback function to shuffle indexes after each epoch"""
        if self.shuffle:
            self.indexes = np.random.permutation(self.indexes)
