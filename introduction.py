"""
File: introduction.py
Author: Djazy Faradj
Created: 10-10-2024
Last modified: 10-10-2024
Description: This is a basic handwritten digit recognition multi-class classification model. This type of approach is one of the simplest for image classification:
A fully-connected neural netweork (also called a perceptron)
"""

import tensorflow as tf
from tensorflow import keras # Keras comes from tensorflow version 2, which is a much higher-level neural network construction API
import matplotlib.pyplot as plt
import numpy as np
import os

#To use GPU memory cautiously, we will set tensorflow option to grow GPU memory allocation when required
physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

def main():
    (x_train, y_train),(x_test,y_test) = keras.datasets.mnist.load_data()

    print (y_train)

    print('Training samples:',len(x_train))
    print('Test samples:',len(x_test))

    print('Tensor size:',x_train[0].shape)
    print('First 10 digits are:', y_train[:10])
    print('Type of data is ',type(x_train))

    print('Min intensity value: ',x_train.min())
    print('Max intensity value: ',x_train.max())

    x_train = x_train.astype(np.float32)/255.0 # Normalizes the value of the image from 0-255 to 0-1
    x_test = x_test.astype(np.float32)/255.0



    model = keras.models.Sequential([
        keras.layers.Flatten(input_shape=(28, 28)), # Define the shape of input and will flatten it, turning it into a layer of 28x28=784 inputs
        keras.layers.Dense(10,activation='softmax') # Softmax will yield a 0 to 1 value also will densify it to a shape of size 10 for output 0 to 9 of labels
    ])

    # Formats the labels so that they can be fitted inside the 10 output long layer of our model
    y_train_onehot = keras.utils.to_categorical(y_train)
    y_test_onehot = keras.utils.to_categorical(y_test)
    """
    First 3 training labels y_train[:3]: [5 0 4]
    One-hot-encoded version y_train_onehot[:3]:
    [[0. 0. 0. 0. 0. 1. 0. 0. 0. 0.]
    [1. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
    [0. 0. 0. 0. 1. 0. 0. 0. 0. 0.]]
    """


    #model.compile(optimizer='sgd',loss='categorical_crossentropy', metrics=['acc']) # Categorical crossentropy is often used when dealing with multi-class classification # Stochastic gradient descent (SGD) is the simplest optimizer, more complex networks use other optimizers like Adam
    model.compile(optimizer=keras.optimizers.SGD(momentum=0.5),loss='categorical_crossentropy',metrics=['acc']) # We manually passed an optimizer object which is called momentum SGD which will average the gradients of loss function over epochs in order to make optimization more smooth
    hist = model.fit(x_train,y_train_onehot,validation_data=(x_test,y_test_onehot), epochs=5, batch_size=128) # Will loop through model.fit() added the parameter validation_data= to determine loss. Note a batch_size has been specified as it is much more efficient to process several samples in one go since GPU computations are easily parallelizable

    # Lets plot our loss and value loss over epochs
    for x in ['loss', 'val_loss']:
        plt.plot(hist.history[x])
    # Lets plot our accuracy and accuracy value over epochs
    for x in ['acc','val_acc']:
        plt.plot(hist.history[x])
    plt.show()

    # Since our model is just one layer and simple, we can easily visualize the weights
    weight_tensor = model.layers[1].weights[0].numpy().reshape(28,28,10)
    fig,ax = plt.subplots(1,10,figsize=(15,4))
    for i in range(10):
        ax[i].imshow(weight_tensor[:,:,i])
        ax[i].axis('off')
    plt.show()
main()