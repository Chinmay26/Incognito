"""
Trainer module for end-to-end training of different segmentation models based on configuration specified in train_params.json

This is inspired by https://github.com/qubvel/segmentation_models/blob/master/examples/binary%20segmentation%20(camvid).ipynb & modified to suit my requirements


"""


import os
import keras
import numpy as np
import matplotlib.pyplot as plt
import json
from model.model import Model
from data.dataset import Dataset
from data.dataset import Dataloder


class Trainer(object):
    """Trainer class to setup training and different model parameters
	"""

    def __init__(self, train_params_file):
        """Initialize based on configuration file train_params.json

			Args
			----
				train_params_file(str): path to configuration file

			Returns
			-------
				None
		"""
        self.full_dir_path = os.path.abspath(os.path.dirname(__file__))
        train_params_full_path = self.full_dir_path + train_params_file

        with open(train_params_full_path) as fp:
            self.train_params_dict = json.load(fp)

        # default train params
        self.saved_model_graph, self.saved_model_weights = None, None
        self.x_train_dir, self.y_train_dir = None, None
        self.x_valid_dir, self.y_valid_dir = None, None
        self.x_test_dir, self.y_test_dir = None, None
        self.h, self.w, self.c = 128, 128, 3

        self.setup_data()

    def setup_data(self):
        """Configure training parameters

			Args
			----
				None

			Returns
			-------
				None
		"""
        dataset_base_dir = self.full_dir_path + self.train_params_dict["data_dir"]
        self.x_train_dir = os.path.join(dataset_base_dir, "train")
        self.y_train_dir = os.path.join(dataset_base_dir, "train_mask")

        self.x_valid_dir = os.path.join(dataset_base_dir, "val")
        self.y_valid_dir = os.path.join(dataset_base_dir, "val_mask")

        self.x_test_dir = os.path.join(dataset_base_dir, "test")
        self.y_test_dir = os.path.join(dataset_base_dir, "test_mask")

        self.h, self.w = (
            self.train_params_dict["model"]["img_size"],
            self.train_params_dict["model"]["img_size"],
        )
        self.batch_size = self.train_params_dict["batch_size"]
        self.epochs = self.train_params_dict["epochs"]

        self.saved_model_graph = (
            self.full_dir_path
            + self.train_params_dict["saved_model_dir"]
            + self.train_params_dict["model"]["encoder"]
            + "_"
            + self.train_params_dict["model"]["type"]
            + "_"
            + str(self.train_params_dict["model"]["img_size"])
            + "_"
            + str(self.train_params_dict["model"]["model_experiment_param"])
            + "_graph.json"
        )

        self.saved_model_weights = (
            self.full_dir_path
            + self.train_params_dict["saved_model_dir"]
            + self.train_params_dict["model"]["encoder"]
            + "_"
            + self.train_params_dict["model"]["type"]
            + "_"
            + str(self.train_params_dict["model"]["img_size"])
            + "_"
            + str(self.train_params_dict["model"]["model_experiment_param"])
            + "_weights.h5"
        )

    def train(self):
        """Train the model based on model and training configurations

			Args
			----
				None

			Returns
			-------
				None
		"""
        print("==== Creating Training dataset =======")
        # Dataset for train images
        train_dataset = Dataset(
            self.x_train_dir, self.y_train_dir, img_size=(self.h, self.w)
        )
        print("==== Creating Validation dataset =======")
        # Dataset for validation images
        valid_dataset = Dataset(
            self.x_valid_dir, self.y_valid_dir, img_size=(self.h, self.w)
        )

        train_dataloader = Dataloder(
            train_dataset, batch_size=self.batch_size, shuffle=True
        )
        valid_dataloader = Dataloder(valid_dataset, batch_size=1, shuffle=False)

        # check shapes for errors
        n_classes = 1
        assert train_dataloader[0][0].shape == (self.batch_size, self.h, self.w, self.c)
        assert train_dataloader[0][1].shape == (
            self.batch_size,
            self.h,
            self.w,
            n_classes,
        )

        # define callbacks for learning rate scheduling and best checkpoints saving
        callbacks = [
            keras.callbacks.ModelCheckpoint(
                self.saved_model_weights,
                save_weights_only=True,
                save_best_only=True,
                mode="min",
            ),
            keras.callbacks.ReduceLROnPlateau(),
        ]
        print("==== Setting up UNET Model =======")
        model = Model(self.train_params_dict)
        complied_model = model.setup_model()
        print("==== Starting Model Training =======")
        history = complied_model.fit_generator(
            train_dataloader,
            steps_per_epoch=len(train_dataloader),
            epochs=self.epochs,
            callbacks=callbacks,
            validation_data=valid_dataloader,
            validation_steps=len(valid_dataloader),
        )
        print("==== Finished Model Training =======")
        print("==== Saving model graph =======")
        # save the model graph
        model_json = complied_model.to_json()
        with open(self.saved_model_graph, "w") as json_file:
            json.dump(model_json, json_file)

        print("==== Finished saving model graph =======")


if __name__ == "__main__":
    params_file = "/./train_params.json"
    trainer = Trainer(params_file)
    trainer.train()
