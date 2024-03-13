"""
Project AI for MIA
Utility functions for the main project
Author: Group 2
"""
# Library imports
import os
import numpy as np

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, GlobalAveragePooling2D, Input, MaxPool2D, UpSampling2D
from tensorflow.keras import utils
from tensorflow.keras.optimizers import Adam, SGD


def get_pcam_generators(base_dir, train_batch_size=32, val_batch_size=32, class_mode='binary', prep_function=None):
    """Function to load data generators."""
    TRAIN_PATH = os.path.join(base_dir, 'train+val', 'train')
    VALID_PATH = os.path.join(base_dir, 'train+val', 'valid')

    RESCALING_FACTOR = 1./255

    # Instantiate data generators with correct preprocessing function
    if prep_function is None:
        datagen = ImageDataGenerator(rescale=RESCALING_FACTOR)
    else:
        datagen = ImageDataGenerator(rescale=RESCALING_FACTOR, preprocessing_function=prep_function)

    train_gen = datagen.flow_from_directory(TRAIN_PATH,
                                            target_size=(96,96),
                                            batch_size=train_batch_size,
                                            class_mode=class_mode)


    val_gen = datagen.flow_from_directory(VALID_PATH,
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
        self.add(Conv2D(first_filters, kernel_size, activation='relu', padding='same', input_shape=(96,96,3)))
        self.add(MaxPool2D(pool_size))
        self.add(Conv2D(second_filters, kernel_size, activation='relu', padding='same'))
        self.add(MaxPool2D(pool_size))

        # Layers replacing the dense layers
        self.add(Conv2D(second_filters, (6,6), activation='relu', padding='valid'))
        self.add(Conv2D(1, (1,1), activation='sigmoid', padding='same'))
        self.add(GlobalAveragePooling2D())

    def compile_cnn(self):
        self.compile(SGD(learning_rate=0.01, momentum=0.95), loss='binary_crossentropy', metrics=['accuracy'])

    def create_autoencoder(self, kernel_size=(3,3), pool_size=(2,2), first_filters=32, second_filters=16):
        # Encoder
        self.add(Conv2D(first_filters, kernel_size, activation='relu', padding='same'))
        self.add(MaxPool2D(pool_size, padding='same'))
        self.add(Conv2D(second_filters, kernel_size, activation='relu', padding='same'))
        self.add(MaxPool2D(pool_size, padding='same'))

        # Decoder
        self.add(Conv2D(second_filters, kernel_size, activation='relu', padding='same'))
        self.add(UpSampling2D(pool_size))
        self.add(Conv2D(first_filters, kernel_size, activation='relu', padding='same'))
        self.add(UpSampling2D(pool_size))
        self.add(Conv2D(3, kernel_size, activation='sigmoid', padding='same'))

    def compile_autoencoder(self):
        self.compile(Adam(learning_rate=0.001), loss='mean_squared_error')


# Written by Constantijn
class Model_transform:
    """Class used as preprocessing function. Responsible for data augmentation with autoencoder model."""

    def __init__(self, ae_model):
        self.ae_model_1 = ae_model

    def model_transform(self, tensor):
        tensor_adjusted = utils.img_to_array(tensor)
        tensor_adjusted = np.array([tensor_adjusted])
        tensor_adjusted_prediction = self.ae_model.predict(tensor_adjusted/255, verbose=None)[0]
        
        return tensor_adjusted_prediction*255
