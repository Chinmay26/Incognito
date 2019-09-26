import os
import numpy as np
import keras
import json
import cv2
from scipy.io import loadmat

class Dataset:
    """
        Basic dataset handler
    """
    
    CLASSES = ['person']
    img_resize = (224,224)
    
    def __init__(
            self, 
            images_dir, 
            masks_dir,
            crop_file,
            classes=None, 
            augmentation=None, 
            preprocessing=None,
    ):
        self.ids = os.listdir(images_dir)
        self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids]
        self.masks_fps = []
        for image_id in self.ids:
            img_name = image_id.split('.')[0]
            mask_name = img_name + "_mask.mat"
            self.masks_fps.append(os.path.join(masks_dir, mask_name))
        
        # convert str names to class values on masks
        self.class_values = [self.CLASSES.index(cls.lower()) for cls in classes]
        
        self.augmentation = augmentation
        self.preprocessing = preprocessing
        self.image_dict = {}
        with open(crop_file) as cp:
            self.image_dict = json.load(cp)
        
    def __getitem__(self, i):
        image = cv2.imread(self.images_fps[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        crop_coords = self.image_dict[self.ids[i]][1]
        assert len(crop_coords) == 4
        y0,y1,x0,x1 = crop_coords[0], crop_coords[1], crop_coords[2], crop_coords[3]
        image = image[ y0:y1, x0:x1, :] # crop out the image 
        image = cv2.resize(image, self.img_resize)
        
        mask_mat = loadmat(self.masks_fps[i])
        mask = mask_mat["mask"].astype(np.uint8)
        mask = cv2.resize(mask, self.img_resize)
        mask = np.expand_dims(mask, axis=2)
        
        return image,mask
    
        
    def __len__(self):
        return len(self.ids)


class Dataloder(keras.utils.Sequence):
    """.
        Image Mask Batch Generator
    """
    
    def __init__(self, dataset, batch_size=1, shuffle=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indexes = np.arange(len(dataset))

        self.on_epoch_end()

    def __getitem__(self, i):
        
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
        """Callback function to shuffle indexes each epoch"""
        if self.shuffle:
            self.indexes = np.random.permutation(self.indexes)
