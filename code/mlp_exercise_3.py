"""
TU/e BME Project Imaging 2021
Simple multiLayer perceptron code for MNIST
Author: Suzanne Wetstein
"""
# disable overly verbose tensorflow logging
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
import tensorflow as tf

# import required packages
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.callbacks import TensorBoard

def redefine_labels(label):
    """ This function is used to redefine labels for the new classes. 
        0 -> "vertical digits": 1, 7
        1 -> "loopy digits": 0, 6, 8, 9
        2 -> "curly digits": 2, 5
        3 -> "other": 3, 4
    """
    if label == 1 or label == 7:
        return 0
    elif label == 2 or label == 5:
        return 2
    elif label == 3 or label == 4:
        return 3
    else:
        return 1

def train_neural_network(nr_hidden_layers=1, nr_neurons=64, nr_epochs=10, activation_function='relu'):
    # load the dataset using the builtin Keras method
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    # redefining the labels of the training set
    for i in range(len(y_train)):
        label = y_train[i]
        new_label = redefine_labels(label)
        y_train[i] = new_label

    # redefining the labels of the testing set
    for i in range(len(y_test)):
        label = y_test[i]
        new_label = redefine_labels(label)
        y_test[i] = new_label

    # derive a validation set from the training set
    # the original training set is split into 
    # new training set (90%) and a validation set (10%)
    X_train, X_val = train_test_split(X_train, test_size=0.10, random_state=101)
    y_train, y_val = train_test_split(y_train, test_size=0.10, random_state=101)

    # the shape of the data matrix is NxHxW, where
    # N is the number of images,
    # H and W are the height and width of the images
    # keras expect the data to have shape NxHxWxC, where
    # C is the channel dimension
    X_train = np.reshape(X_train, (-1,28,28,1)) 
    X_val = np.reshape(X_val, (-1,28,28,1))
    X_test = np.reshape(X_test, (-1,28,28,1))

    # convert the datatype to float32
    X_train = X_train.astype('float32')
    X_val = X_val.astype('float32')
    X_test = X_test.astype('float32')

    # normalize our data values to the range [0,1]
    X_train /= 255
    X_val /= 255
    X_test /= 255

    # convert 1D class arrays to 4D class matrices
    y_train = to_categorical(y_train, 4)
    y_val = to_categorical(y_val, 4)
    y_test = to_categorical(y_test, 4)

    model = Sequential()
    # flatten the 28x28x1 pixel input images to a row of pixels (a 1D-array)
    model.add(Flatten(input_shape=(28,28,1))) 
    # fully connected hidden layers with chosen amount of neurons and ReLU nonlinearity
    for i in range(nr_hidden_layers):
        model.add(Dense(nr_neurons, activation=activation_function))
    # output layer with 4 nodes (one for each class) and softmax nonlinearity
    model.add(Dense(4, activation='softmax')) 

    # compile the model
    model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

    # custom model name based on chosen number of hidden layers and neurons
    model_name = f"mlp_ex_3_{nr_hidden_layers}_layers_{nr_neurons}_neurons_{nr_epochs}_epochs_{activation_function}"

    # create a way to monitor our model in Tensorboard
    tensorboard = TensorBoard("logs/" + model_name)

    # train the model
    model.fit(X_train, y_train, batch_size=32, epochs=nr_epochs, verbose=1, validation_data=(X_val, y_val), callbacks=[tensorboard])

    score = model.evaluate(X_test, y_test, verbose=0)

    print(f"{model_name} Loss: ", score[0])
    print(f"{model_name} Accuracy: ", score[1])

train_neural_network()