"""
Project AI for MIA
Utility functions for the main project
Author: Group 2
"""
# Library imports
import os
import random
import numpy as np

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, GlobalAveragePooling2D, Input, MaxPool2D, UpSampling2D
from tensorflow.keras import utils
from tensorflow.keras.optimizers import Adam, SGD, Adadelta
from tensorflow.keras import backend



def get_pcam_generators(base_dir, train_batch_size=32, val_batch_size=32, class_mode="binary", prep_function=None):
    """Function to load data generators."""
    
    TRAIN_PATH = os.path.join(base_dir, "train+val", "train")
    VALID_PATH = os.path.join(base_dir, "train+val", "valid")

    RESCALING_FACTOR = 1./255

    # Instantiate data generators with correct preprocessing function
    datagen_train = ImageDataGenerator(rescale=RESCALING_FACTOR, preprocessing_function=prep_function)
    datagen_val = ImageDataGenerator(rescale=RESCALING_FACTOR)

    train_gen = datagen_train.flow_from_directory(TRAIN_PATH,
                                                  target_size=(96,96),
                                                  batch_size=train_batch_size,
                                                  class_mode=class_mode)

    val_gen = datagen_val.flow_from_directory(VALID_PATH,
                                              target_size=(96,96),
                                              batch_size=val_batch_size,
                                              class_mode=class_mode,
                                              shuffle=False)
     
    return train_gen, val_gen


class Model_architecture(Sequential):
    """Class to load model structures."""

    def __init__(self):
        super().__init__()
        self.add(Input(shape=(96,96,3)))

    def create_cnn(self, kernel_size=(3,3), pool_size=(4,4), first_filters=32, second_filters=64):
        # First set of convolutional layers
        self.add(Conv2D(first_filters, kernel_size, activation="relu", padding="same", input_shape=(96,96,3)))
        self.add(MaxPool2D(pool_size))
        self.add(Conv2D(second_filters, kernel_size, activation="relu", padding="same"))
        self.add(MaxPool2D(pool_size))

        # Layers replacing the dense layers
        self.add(Conv2D(second_filters, (6,6), activation="relu", padding="valid"))
        self.add(Conv2D(1, (1,1), activation="sigmoid", padding="same"))
        self.add(GlobalAveragePooling2D())

    def compile_cnn(self, learning_rate=0.001):
        self.compile(Adam(learning_rate=learning_rate), loss="binary_crossentropy", metrics=["accuracy"])
               
    def create_autoencoder(self, kernel_size=(3,3), pool_size=(2,2), first_filters=32, second_filters=16):
        # Encoder
        self.add(Conv2D(first_filters, kernel_size, activation="relu", padding="same"))
        self.add(MaxPool2D(pool_size, padding="same"))
        self.add(Conv2D(second_filters, kernel_size, activation="relu", padding="same"))
        self.add(MaxPool2D(pool_size, padding="same"))

        # Decoder
        self.add(Conv2D(second_filters, kernel_size, activation="relu", padding="same"))
        self.add(UpSampling2D(pool_size))
        self.add(Conv2D(first_filters, kernel_size, activation="relu", padding="same"))
        self.add(UpSampling2D(pool_size))
        self.add(Conv2D(3, kernel_size, activation="sigmoid", padding="same"))

    def compile_autoencoder(self, learning_rate=0.001):
        self.compile(Adam(learning_rate=learning_rate), loss="mean_squared_error")


# Written by Constantijn
class Model_transform:
    """Class used as preprocessing function. Responsible for data augmentation with autoencoder model."""

    def __init__(self, ae_model, augmentation_factor=1):
        """
        Args: 
        ae_model: autoencoder model
        augmentation_factor: float between 0 and 1, determines the amount of data augmentation. 1 being only augmented data and 0 beging only original data. 
        """
        self.ae_model = ae_model
        self.augmentation_factor = augmentation_factor
        self.choices = [0 , 1]
        self.weights = [1-augmentation_factor, augmentation_factor]

    def model_transform(self, tensor):
        if random.choices(self.choices, weights=self.weights)[0]:
            tensor_adjusted = utils.img_to_array(tensor)
            tensor_adjusted = np.array([tensor_adjusted])
            tensor_adjusted_prediction = self.ae_model.predict(tensor_adjusted/255, verbose=None)[0]
            backend.clear_session()
            return tensor_adjusted_prediction*255
        return tensor
