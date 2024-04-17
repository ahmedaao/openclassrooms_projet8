import os
import numpy as np
from tensorflow.keras.utils import Sequence
from tensorflow.keras.preprocessing import image


dataset_path = "/home/hao/repositories/design-autonomous-car/data/processed/dataset/"
train_images = dataset_path + "train_images/"
train_masks = dataset_path + "train_masks/"
val_images = dataset_path + "val_images/"
val_masks = dataset_path + "val_masks/"

train_image_list = sorted(os.listdir(train_images))
train_mask_list = sorted(os.listdir(train_masks))
val_image_list = sorted(os.listdir(val_images))
val_mask_list = sorted(os.listdir(val_masks))

cats = {'void': [0, 1, 2, 3, 4, 5, 6],
        'flat': [7, 8, 9, 10],
        'construction': [11, 12, 13, 14, 15, 16],
        'object': [17, 18, 19, 20],
        'nature': [21, 22],
        'sky': [23],
        'human': [24, 25],
        'vehicle': [26, 27, 28, 29, 30, 31, 32, 33, -1]}


class DataGenerator(Sequence):
    """Processing using batch method

    Args:
        Sequence (class): Parent keras class to process by batch
    """
    def __init__(self, x_set, y_set, input_height,
                 input_width, batch_size):
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size
        self.input_height = input_height
        self.input_width = input_width

    def __len__(self):
        return int(np.ceil(len(self.x) / float(self.batch_size)))

    def __getitem__(self, idx):
        idx = np.random.randint(0, len(train_image_list), self.batch_size)
        batch_x, batch_y = [], []
        drawn = 0
        for i in idx:
            _image = image.img_to_array(image.load_img(f'{train_images}/{train_image_list[i]}', target_size=(self.input_height, self.input_width)))/255.
            img = image.img_to_array(image.load_img(f'{train_masks}/{train_mask_list[i]}', grayscale=True, target_size=(self.input_height, self.input_width)))
            labels = np.unique(img)
            if len(labels) < 3:
                idx = np.random.randint(0, len(train_image_list),
                                        self.batch_size-drawn)
                continue
            img = np.squeeze(img)
            mask = np.zeros((img.shape[0], img.shape[1], 8))
            for i in range(-1, 34):
                if i in cats['void']:
                    mask[:, :, 0] = np.logical_or(mask[:, :, 0], (img == i))
                elif i in cats['flat']:
                    mask[:, :, 1] = np.logical_or(mask[:, :, 1], (img == i))
                elif i in cats['construction']:
                    mask[:, :, 2] = np.logical_or(mask[:, :, 2], (img == i))
                elif i in cats['object']:
                    mask[:, :, 3] = np.logical_or(mask[:, :, 3], (img == i))
                elif i in cats['nature']:
                    mask[:, :, 4] = np.logical_or(mask[:, :, 4], (img == i))
                elif i in cats['sky']:
                    mask[:, :, 5] = np.logical_or(mask[:, :, 5], (img == i))
                elif i in cats['human']:
                    mask[:, :, 6] = np.logical_or(mask[:, :, 6], (img == i))
                elif i in cats['vehicle']:
                    mask[:, :, 7] = np.logical_or(mask[:, :, 7], (img == i))
            #mask = np.resize(mask,(resize_input_height*resize_input_width, 8))
            batch_y.append(mask)
            batch_x.append(_image)
            drawn += 1
        return np.array(batch_x), np.array(batch_y)
