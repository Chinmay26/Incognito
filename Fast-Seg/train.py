import os
import keras
import numpy as np
import matplotlib.pyplot as plt
import json
from model.model import Model
from data.dataset import Dataset
from data.dataset import Dataloder


# This file is  based on examples given at https://github.com/qubvel/segmentation_models/blob/master/examples/binary%20segmentation%20(camvid).ipynb & modified to suit my requirements

# TO-DO change these hardcoded paths & load from a json params file
DATA_DIR = "/home/chinmay/Code/datasets/processed"
x_train_dir = os.path.join(DATA_DIR, "train")
y_train_dir = os.path.join(DATA_DIR, "trainannot")

x_valid_dir = os.path.join(DATA_DIR, "val")
y_valid_dir = os.path.join(DATA_DIR, "valannot")

x_test_dir = os.path.join(DATA_DIR, "test")
y_test_dir = os.path.join(DATA_DIR, "testannot")

crop_file = os.path.join(DATA_DIR, "./image_mask.json")


def train():

    BACKBONE = "mobilenetv2"
    BATCH_SIZE = 4
    CLASSES = ["person"]
    LR = 0.0001
    EPOCHS = 5
    h, w, c = 224, 224, 3
    n_classes = 1 if len(CLASSES) == 1 else (len(CLASSES) + 1)

    # Dataset for train images
    train_dataset = Dataset(x_train_dir, y_train_dir, crop_file, classes=CLASSES)

    # Dataset for validation images
    valid_dataset = Dataset(x_valid_dir, y_valid_dir, crop_file, classes=CLASSES)

    train_dataloader = Dataloder(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    valid_dataloader = Dataloder(valid_dataset, batch_size=1, shuffle=False)

    # check shapes for errors
    assert train_dataloader[0][0].shape == (BATCH_SIZE, h, w, c)
    assert train_dataloader[0][1].shape == (BATCH_SIZE, h, w, n_classes)

    # define callbacks for learning rate scheduling and best checkpoints saving
    # TO-DO change hardcoded path
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            "./UnetMNV2/best_model.h5",
            save_weights_only=True,
            save_best_only=True,
            mode="min",
        ),
        keras.callbacks.ReduceLROnPlateau(),
    ]

    EPOCHS = 2

    model = Model().setup_model()
    history = model.fit_generator(
        train_dataloader,
        steps_per_epoch=len(train_dataloader),
        epochs=EPOCHS,
        callbacks=callbacks,
        validation_data=valid_dataloader,
        validation_steps=len(valid_dataloader),
    )


if __name__ == "__main__":
    train()
