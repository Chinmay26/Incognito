"""
Base Model for different segmentation backbone encoders and decoders

This is inspired by https://github.com/qubvel/segmentation_models/blob/master/examples/binary%20segmentation%20(camvid).ipynb & modified to suit my requirements

"""

import segmentation_models as sm
import keras
import os
import json


class Model(object):
    EXPERIMENTS_BASE_FILE_PATH = "/../experiments/base_model/model_params.json"
    EXPERIMENTS_EFF_FILE_PATH = "/../experiments/efficient_net/model_params.json"
    METRIC_THRESHOLD = 0.5

    def __init__(self, train_params):
        """Initialize based on configuration specified in model_params.json

			Args
			----
				train_params(dict): training parameter configiration

			Returns
			------
				None
		"""
        full_dir_path = os.path.abspath(os.path.dirname(__file__))
        base_model_params_full_path = full_dir_path + self.EXPERIMENTS_BASE_FILE_PATH
        eff_model_params_full_path = full_dir_path + self.EXPERIMENTS_EFF_FILE_PATH

        with open(base_model_params_full_path) as fp:
            self.base_model_params = json.load(fp)

        with open(eff_model_params_full_path) as fp:
            self.eff_model_params = json.load(fp)

        # default model params
        self.learning_rate = 0.0001
        self.h, self.w, self.c = 224, 224, 3
        self.classes = ["person"]
        self.n_classes = 1 if len(self.classes) == 1 else (len(self.classes) + 1)
        self.train_params = train_params
        self.backbone = "mobilenetv2"
        self.total_loss = sm.losses.DiceLoss()

        self.setup_model_params()

    def setup_model_params(self):
        """Initalize model params based on model configuration file

			Args
			----
				None

			Returns
			------
				None
		"""
        experiment_params = {}
        if self.train_params:
            model_type = self.train_params["model"]["type"]
            img_size = str(self.train_params["model"]["img_size"])
            experiment = self.train_params["model"]["model_experiment_param"]
            if self.train_params["model"]["encoder"] == "efficient_net":
                experiment_params = self.eff_model_params["efficient_net"][model_type][
                    img_size
                ][experiment]
                self.backbone = self.eff_model_params["efficient_net"][model_type][
                    "backbone"
                ]
            elif self.train_params["model"]["encoder"] == "base":
                experiment_params = self.base_model_params[model_type][img_size][
                    experiment
                ]

        self.h, self.w, self.c = (
            experiment_params["image_h"],
            experiment_params["image_w"],
            experiment_params["channels"],
        )
        self.learning_rate = experiment_params["learning_rate"]
        if "focal_loss" in experiment_params["loss"]:
            dice_loss = sm.losses.DiceLoss()
            focal_loss = (
                sm.losses.BinaryFocalLoss()
                if self.n_classes == 1
                else sm.losses.CategoricalFocalLoss()
            )
            self.total_loss = dice_loss + (1 * focal_loss)
        self.activation = experiment_params["activation"]

    def setup_model(self):
        """Setup model with optimizer, loss and metrics

			Args
			----
				None

			Returns
			-------
				model(segmentation_models.model): Compiled segmentation model
		"""
        # create model
        model = sm.Unet(
            self.backbone,
            classes=self.n_classes,
            activation=self.activation,
            input_shape=(self.h, self.w, self.c),
        )

        # optimizer
        optim = keras.optimizers.Adam(self.learning_rate)

        # metrics
        metrics = [
            sm.metrics.IOUScore(threshold=self.METRIC_THRESHOLD),
            sm.metrics.FScore(threshold=self.METRIC_THRESHOLD),
        ]

        # compile keras model with defined optimozer, loss and metrics
        model.compile(optim, self.total_loss, metrics)

        return model
