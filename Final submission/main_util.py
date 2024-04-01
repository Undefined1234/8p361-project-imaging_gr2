"""
Project AI for MIA
Utility functions for the main project
Author: Group 2
"""
# Library imports
import os
import random
import numpy as np

from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, GlobalAveragePooling2D, Input, MaxPool2D, UpSampling2D
from tensorflow.keras import utils
from tensorflow.keras.optimizers import Adam, SGD, Adadelta
from tensorflow.keras import backend


def get_pcam_generators(base_dir, train_batch_size=16, val_batch_size=16, class_mode="binary", prep_function=None):
    """Function to load data generators."""
    
    TRAIN_PATH = os.path.join(base_dir, "train+val", "train")
    VALID_PATH = os.path.join(base_dir, "train+val", "valid")

    RESCALING_FACTOR = 1./255

    # Instantiate training data generator with preprocessing function
    datagen_train = ImageDataGenerator(rescale=RESCALING_FACTOR, preprocessing_function=prep_function)

    # Instantiate validation data generator without preprocessing function
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
    """Class to load all necessary model structures."""

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


class Model_transform:
    """Class used as preprocessing function in data generators. Responsible for data augmentation with autoencoder model."""

    def __init__(self, ae_model, augmentation_factor=1):
        """
        Args
        ae_model:               autoencoder model.
        augmentation_factor:    float between 0 and 1, determines the amount of data augmentation. 
                                1 being only augmented data and 0 beging only original data. 
        """
        self.ae_model = ae_model
        self.augmentation_factor = augmentation_factor
        self.choices = [0 , 1]
        self.weights = [1-augmentation_factor, augmentation_factor]

    def model_transform(self, tensor):
        """
        Args
        tensor: variable for the image that is augmented.
        """
        # Applying a weighted randomizer
        if random.choices(self.choices, weights=self.weights)[0]:
            # Additional preprocessing in order to make a prediction
            tensor_adjusted = utils.img_to_array(tensor)
            tensor_adjusted = np.array([tensor_adjusted])
            tensor_adjusted_prediction = self.ae_model.predict(tensor_adjusted/255, verbose=None)[0]

            # Clearing backend to prevent memory overflow
            backend.clear_session()

            return tensor_adjusted_prediction*255
        
        return tensor
    

def evaluation_pcam17(model, patch_folder_path, testing_metadata):
    """
    Args
    model:              model object to perform predictions.
    patch_folder_path:  string containing the filepath to the patches folder.
    testing_metadata:   numpy array containing the metadata of the testing set with the following columns:
                        index, patient nr, node nr, x-coordinate, y-coordinate, label, slide nr, split nr
    """
    true_labels = []
    pred_labels = []
    percent_complete = 0
    for i in range(len(testing_metadata)):
        # Saving values from the metadata inside variables
        patient_nr = testing_metadata[i,1]
        node_nr = testing_metadata[i,2]
        x_coord = testing_metadata[i,3]
        y_coord = testing_metadata[i,4]

        # Different filepath to the patch based on one- or two-digit patient nr  
        if patient_nr < 10:
             patch_path = f"{patch_folder_path}/patient_00{patient_nr}_node_{node_nr}/patch_patient_00{patient_nr}_node_{node_nr}_x_{x_coord}_y_{y_coord}.png"
        else:
            patch_path = f"{patch_folder_path}/patient_0{patient_nr}_node_{node_nr}/patch_patient_0{patient_nr}_node_{node_nr}_x_{x_coord}_y_{y_coord}.png"

        # Preprocessing of the image before prediction
        img = img_to_array(load_img(patch_path, target_size=(96,96)))
        img = np.array([img])

        # Predicting labels using model and saving predicted labels   
        pred_label = model.predict(img/255, verbose=None)
        pred_labels.append(pred_label[0][0])

        # Saving true labels
        true_label = testing_metadata[i,5]
        true_labels.append(true_label)

        # Clearing memory and progression message for each 10% of the testing set completed
        backend.clear_session()
        if i%(len(testing_metadata)//10) == 0:
            print(f"Progress: {percent_complete}% done")
            percent_complete += 10

    return true_labels, pred_labels
