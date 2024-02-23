import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Conv2D
from tensorflow.keras.callbacks import TensorBoard

#####################Preprocessing#####################

(X_train, y_train), (X_test, y_test) = mnist.load_data() #loading standard dataset 
X_train, X_val = train_test_split(X_train, test_size=0.10, random_state=101) #splitting the data into training and testing sets
y_train, y_val = train_test_split(y_train, test_size=0.10, random_state=101)

# the shape of the data matrix is NxHxW, where
# N is the number of images,
# H and W are the height and width of the images
# keras expect the data to have shape NxHxWxC, where
# C is the channel dimension
X_train = np.reshape(X_train, (-1,28,28,1)) 
X_val = np.reshape(X_val, (-1,28,28,1))
X_test = np.reshape(X_test, (-1,28,28,1))

#Set the datatype to float32
X_train = X_train.astype('float32')
X_val = X_val.astype('float32')
X_test = X_test.astype('float32')

#Normalize data between 0 and 1
X_train /= 255
X_val /= 255
X_test /= 255

new_classification = dict({"0":[1,7], "1":[0,6,8,9], "2":[2,5], "3":[3,4]}) #new classification of the numbers, given by the exercise 
def converter(number):
    """This function converts the numbers into the given classification. 
    Input: number (int) - the number to be converted
    Output: int - the new classification of the number"""
    for key in new_classification:
        if number in new_classification[key]:
            return int(key)

y_train = np.array([converter(y) for y in y_train]) #apply converter on the training set
y_val = np.array([converter(y) for y in y_val])
y_test = np.array([converter(y) for y in y_test])

y_train = to_categorical(y_train, 4)
y_val = to_categorical(y_val, 4)
y_test = to_categorical(y_test, 4)

#####################Modelling#####################

model8 = Sequential() #initializing the model

"""Two convolutional filters are added to the model. It is known that convolutional fitlers work well when classifiying images."""
model8.add(Conv2D(filters=16, kernel_size=(3,3), activation='relu', input_shape=(28,28,1)))
model8.add(Conv2D(filters=16, padding='valid' , kernel_size=(3,3), activation='relu'))

model8.add(Flatten()) #flatten the output of the second convolutional filter

model8.add(Dense(64, activation='relu')) #add a fully connected layer with 64 neurons

model8.add(Dense(4, activation='softmax')) #add a fully connected layer with 4 neurons, as there are 4 classes. In this case, the activation function is softmax, as it is a classification problem.

model8.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy']) #Compile the model 

model_name="my_eight_model" #name of the model

tensorboard = TensorBoard("../logs/{}".format(model_name)) #initialize tensorboard

model8.fit(X_train, y_train, batch_size=32, epochs=10, verbose=1, validation_data=(X_val, y_val), callbacks=[tensorboard]) #fit the training data to the model

#Postprocessing
score = model8.evaluate(X_test, y_test, verbose=0) #evaluate the model on the test set

print("Loss: ",score[0])
print("Accuracy: ",score[1])
