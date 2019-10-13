"""Create a mashup Portrait-Indoor dataset by embedding Flickr Potrait Images onto different backgrounds from MIT Indoor dataset.

   Using 15 indoor categories from MIT Indoor dataset, create a new dataset by embedding different transformations (random left aligned, random right aligned) of Flickr potrait images.

   Split the dataset into train ~ 10k images, val ~ 1.5k images, test ~ 1.5k images

"""


import json
import cv2
import os
import glob
import random
import skimage
import numpy as np
from scipy.io import loadmat
from mashup_helper import portrait_indoor_embed, show_images
from tqdm import tqdm


class PotraitIndoorDataset(object):
    """Creates a mashup dataset (embeds Flickr dataset potrait images onto MIT dataset indoor images)
    """

    MASK_FORMAT = ".mat"
    PNG_FORMAT = ".png"
    JPG_FORMAT = ".jpg"
    TRAIN_SPLIT = 0.7
    VAL_SPLIT = 0.8

    def __init__(self, params_file):
        """Initialize the params json file

           Args
           ----
               params_file(str): path to params file

        """

        with open(params_file) as fp:
            params_dict = json.load(fp)

        self.potrait_image_dir = params_dict["potrait_image_dir"]
        self.mask_dir = params_dict["mask_dir"]
        self.indoor_image_dir = params_dict["indoor_image_dir"]
        self.agumented_dir = os.path.abspath(
            os.path.dirname(params_dict["agumented_dir"])
        )
        with open(params_dict["mask_crop_file_path"]) as fp:
            self.image_dict = json.load(fp)

        random.seed(42)
        self.image_h, self.image_w = 600, 800
        self.train_potrait_images, self.val_potrait_images = [], []
        self.indoor_train, self.indoor_test, self.indoor_val = [], [], []

    def process_potrait(self, img_path):
    	"""Processes Flickr image potrait

    	   Args
    	   ----
    	       img_path(str): file_path of the image

    	   Returns
    	   -------
    	       rgb(np.array): image resized and cropped

    	"""
        bgr = cv2.imread(img_path)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        img_file_name = os.path.basename(img_path)
        crop_coords = self.image_dict[img_file_name][1]
        assert len(crop_coords) == 4
        y0, y1, x0, x1 = crop_coords[0], crop_coords[1], crop_coords[2], crop_coords[3]
        rgb = rgb[y0:y1, x0:x1, :]
        rgb = cv2.resize(rgb, (self.image_h, self.image_w))
        return rgb

    def process_potrait_mask(self, mask_path):
        """Load masks from Flickr dataset which are stored as MATLAB files
		"""
        mat = loadmat(mask_path)
        return mat["mask"].astype(np.uint8)

    def setup_dataset(self):

        indoor_files = [
            f for f in glob.glob(self.indoor_image_dir + "**/*.jpg", recursive=True)
        ]
        l = len(indoor_files)
        # print(indoor_files[0])
        random.shuffle(indoor_files)
        # print(indoor_files[0])

        self.train_potrait_images = [
            f
            for f in glob.glob(
                self.potrait_image_dir + "train/" + "*.jpg", recursive=True
            )
        ]
        self.val_potrait_images = [
            f
            for f in glob.glob(
                self.potrait_image_dir + "val/" + "*.jpg", recursive=True
            )
        ]
        random.shuffle(self.train_potrait_images)
        random.shuffle(self.val_potrait_images)

        train = int(self.TRAIN_SPLIT * l)
        val = int(self.VAL_SPLIT * l)
        # print(train_potrait_images[0], val_potrait_images[0], train, val)
        self.indoor_train = indoor_files[:train]
        self.indoor_val = indoor_files[train:val]
        self.indoor_test = indoor_files[val:]

    def create_dataset(self):
        self.setup_dataset()

        print(
            "Creating Validation dataset by embedding Portrait images onto Indoor scene images"
        )
        self.create_val_dataset()
        print("Finished Creating Training dataset")
        print("Creating Test dataset")
        self.create_test_dataset()
        print("Finished Creating Test dataset")

        print(
            "Creating Training dataset by embedding Portrait images onto Indoor scene images"
        )
        self.create_train_dataset()
        print("Finished Creating Training dataset")

    def create_train_dataset(self):
        """Create Training split of the new dataset

		For Train:
		    Potrait images: 800
		    Indoor images : 475
		    
		    Get 4 indoor images at random from indoor images for one potrait & generate transforms
		    
		    Total images = 800 * 4 indoor * 3 transformations = 9600
		    

		"""

        img_data = self.train_potrait_images
        img_data_dir = "/train/"
        mask_data_dir = "/train_mask/"

        args = ["none", "left_random", "right_random"]
        exceptions = []
        for potrait_img_path in tqdm(self.train_potrait_images):
            for i in range(4):
                indoor_img_path = self.indoor_train.pop()
                indoor_img = skimage.io.imread(indoor_img_path)
                potrait_img = self.process_potrait(potrait_img_path)

                # process potrait
                img_fullname = os.path.basename(potrait_img_path)
                img_name, extension = os.path.splitext(img_fullname)
                expected_mask_file_name = img_name + "_mask" + self.MASK_FORMAT
                expected_mask_path = os.path.join(
                    self.mask_dir, expected_mask_file_name
                )
                potrait_mask = self.process_potrait_mask(expected_mask_path)

                # print(expected_mask_path, potrait_img_path, indoor_img_path)

                indoor_img_fullname = os.path.basename(indoor_img_path)
                indoor_img_name, extension = os.path.splitext(indoor_img_fullname)

                potrait_img_fullname = os.path.basename(potrait_img_path)
                potrait_img_name, extension = os.path.splitext(potrait_img_fullname)

                aug_img_name = (
                    self.agumented_dir
                    + img_data_dir
                    + potrait_img_name
                    + "_"
                    + indoor_img_name
                    + "_"
                )
                mask_img_name = (
                    self.agumented_dir
                    + mask_data_dir
                    + potrait_img_name
                    + "_"
                    + indoor_img_name
                    + "_"
                )
                print(aug_img_name, mask_img_name)

                try:
                    for a in args:
                        embedded_image, mask = portrait_indoor_embed(
                            portrait_image_input=potrait_img,
                            portrait_mask_input=potrait_mask,
                            indoor_image_input=indoor_img,
                            shift_arg=a,
                            random_affine=True,
                            intensity=1.0,
                        )

                        print(
                            embedded_image.shape,
                            embedded_image[0][0][0],
                            aug_img_name + a + self.JPG_FORMAT,
                        )
                        skimage.io.imsave(
                            aug_img_name + a + self.JPG_FORMAT, embedded_image
                        )
                        skimage.io.imsave(mask_img_name + a + self.PNG_FORMAT, mask)
                except:
                    exceptions.append(
                        [expected_mask_path, potrait_img_path, indoor_img_path]
                    )
                    continue

    def create_val_dataset(self):
        """Create Validation split of the new dataset

		For Val:
		    Potrait images: 544
		    Indoor images : 475
		    
		    Get 1 indoor images + potrait 
		    
		    Total images = 544 * 1 indoor * 3 transformations = 1632
		    

		"""

        img_data_dir = "/val/"
        mask_data_dir = "/val_mask/"

        args = ["none", "left_random", "right_random"]
        exceptions = []
        for potrait_img_path in tqdm(self.val_potrait_images):
            indoor_img_path = self.indoor_val[
                random.randint(0, len(self.indoor_val) - 1)
            ]
            indoor_img = skimage.io.imread(indoor_img_path)
            potrait_img = self.process_potrait(potrait_img_path)

            # process potrait
            img_fullname = os.path.basename(potrait_img_path)
            img_name, extension = os.path.splitext(img_fullname)
            expected_mask_file_name = img_name + "_mask" + self.MASK_FORMAT
            expected_mask_path = os.path.join(self.mask_dir, expected_mask_file_name)
            potrait_mask = self.process_potrait_mask(expected_mask_path)

            # print(expected_mask_path, potrait_img_path, indoor_img_path)

            indoor_img_fullname = os.path.basename(indoor_img_path)
            indoor_img_name, extension = os.path.splitext(indoor_img_fullname)

            potrait_img_fullname = os.path.basename(potrait_img_path)
            potrait_img_name, extension = os.path.splitext(potrait_img_fullname)

            aug_img_name = (
                self.agumented_dir
                + img_data_dir
                + potrait_img_name
                + "_"
                + indoor_img_name
                + "_"
            )
            mask_img_name = (
                self.agumented_dir
                + mask_data_dir
                + potrait_img_name
                + "_"
                + indoor_img_name
                + "_"
            )

            try:
                for a in args:
                    embedded_image, mask = portrait_indoor_embed(
                        portrait_image_input=potrait_img,
                        portrait_mask_input=potrait_mask,
                        indoor_image_input=indoor_img,
                        shift_arg=a,
                        random_affine=True,
                        intensity=1.0,
                    )

                    skimage.io.imsave(
                        aug_img_name + a + self.JPG_FORMAT, embedded_image
                    )
                    skimage.io.imsave(mask_img_name + a + self.PNG_FORMAT, mask)
            except:
                exceptions.append(
                    [expected_mask_path, potrait_img_path, indoor_img_path]
                )
                continue

    def create_test_dataset(self):
        """Create Test split of the new dataset

		For Test:
		    Potrait images: 238
		    Indoor images : 950
		    
		    Get 1 indoor images + potrait 
		    
		    Total images = 238 * 2 indoor * 3 transformations = 1428
		    

		"""
        img_data_dir = "/test/"
        mask_data_dir = "/test_mask/"
        args = ["none", "left_random", "right_random"]
        exceptions = []
        test_potrait_images = [
            f for f in glob.glob(potrait_image_dir + "test/" + "*.jpg", recursive=True)
        ]
        for potrait_img_path in tqdm(test_potrait_images):
            for i in range(2):
                indoor_img_path = self.indoor_test[
                    random.randint(0, len(self.indoor_test) - 1)
                ]
                indoor_img = skimage.io.imread(indoor_img_path)
                potrait_img = self.process_potrait(potrait_img_path)

                # process potrait
                img_fullname = os.path.basename(potrait_img_path)
                img_name, extension = os.path.splitext(img_fullname)
                expected_mask_file_name = img_name + "_mask" + self.MASK_FORMAT
                expected_mask_path = os.path.join(
                    self.mask_dir, expected_mask_file_name
                )
                potrait_mask = process_potrait_mask(expected_mask_path)

                print(expected_mask_path, potrait_img_path, indoor_img_path)

                indoor_img_fullname = os.path.basename(indoor_img_path)
                indoor_img_name, extension = os.path.splitext(indoor_img_fullname)

                potrait_img_fullname = os.path.basename(potrait_img_path)
                potrait_img_name, extension = os.path.splitext(potrait_img_fullname)

                aug_img_name = (
                    self.agumented_dir
                    + img_data_dir
                    + potrait_img_name
                    + "_"
                    + indoor_img_name
                    + "_"
                )
                mask_img_name = (
                    self.agumented_dir
                    + mask_data_dir
                    + potrait_img_name
                    + "_"
                    + indoor_img_name
                    + "_"
                )

                try:
                    for a in args:
                        embedded_image, mask = portrait_indoor_embed(
                            portrait_image_input=potrait_img,
                            portrait_mask_input=potrait_mask,
                            indoor_image_input=indoor_img,
                            shift_arg=a,
                            random_affine=True,
                            intensity=1.0,
                        )

                        skimage.io.imsave(
                            aug_img_name + a + self.JPG_FORMAT, embedded_image
                        )
                        skimage.io.imsave(mask_img_name + a + self.PNG_FORMAT, mask)
                except:
                    exceptions.append(
                        [expected_mask_path, potrait_img_path, indoor_img_path]
                    )
                    continue


if __name__ == "__main__":
    params_file = "./params.json"
    mashup = PotraitIndoorDataset(params_file)
    mashup.create_dataset()
