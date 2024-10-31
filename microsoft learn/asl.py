"""
File: asl.py
Author: Djazy Faradj
Created: 10-10-2024
Last modified: 10-10-2024
Description: asl.py is a script which trains models on ASL alphabet pictures.
"""

from constants import *
from PIL import Image
import os
import tensorflow as tf
from typing import Tuple
import numpy as np


labels_map = { # Assigns a number to each label
    0 : 'A',
    1 : 'B',
    2 : 'C',
    3 : 'D',
    4 : 'E',
    5 : 'F',
    6 : 'G',
    7 : 'H',
    8 : 'I',
    9 : 'J',
    10 : 'K',
    11 : 'L',
    12 : 'M',
    13 : 'N',
    14 : 'O',
    15 : 'P',
    16 : 'Q',
    17 : 'R',
    18 : 'S',
    19 : 'T',
    20 : 'U',
    21 : 'V',
    22 : 'W',
    23 : 'X',
    24 : 'Y',
    25 : 'Z',
    26 : 'space',
    27 : 'del',
    28 : 'nothing'
}

training_dataset_size = 1*29 # There are 3000 training images per the 29 different labels


class NeuralNetwork(tf.keras.Model):
  def __init__(self):
    super(NeuralNetwork, self).__init__()
    self.sequence = tf.keras.Sequential([
      tf.keras.layers.Flatten(input_shape=(200, 200)),
      tf.keras.layers.Dense(20, activation='relu'),
      tf.keras.layers.Dense(10)
    ])

  def call(self, x: tf.Tensor) -> tf.Tensor:
    y_prime = self.sequence(x)
    return y_prime

def load_images(training_dataset_size : int): # Will load both images and labels data[0] = images | data[1] = labels
    directory = os.fsencode("datasets/asl_alphabet_test/asl_alphabet_test") # Goes over each letter folder inside the training folder
    i = 0
    images = np.zeros(training_dataset_size, dtype=tuple)
    labels = np.zeros(len(labels_map))
    for letter_folder in os.listdir(directory):
        filename = os.fsdecode(letter_folder)
        image_directory = os.fsencode(directory+b"/"+letter_folder)
        for letter_image in os.listdir(image_directory): # Loops through every image present inside the dataset
                image_filename = os.fsdecode(letter_image)
                im = Image.open(directory.decode("utf-8")+"/"+filename+"/"+image_filename)
                im = im.convert("L") # Converts training image into black and white
                img_data = np.asarray(im.getdata())
                img_data.shape = (200, 200)
                print(img_data)
                images[i] = img_data
                if (filename == labels_map.get(0)): labels[i] = 0
                elif (filename == labels_map.get(1)): labels[i] = 1
                elif (filename == labels_map.get(2)): labels[i] = 2
                elif (filename == labels_map.get(3)): labels[i] = 3
                elif (filename == labels_map.get(4)): labels[i] = 4
                elif (filename == labels_map.get(5)): labels[i] = 5
                elif (filename == labels_map.get(6)): labels[i] = 6
                elif (filename == labels_map.get(7)): labels[i] = 7
                elif (filename == labels_map.get(8)): labels[i] = 8
                elif (filename == labels_map.get(9)): labels[i] = 9
                elif (filename == labels_map.get(10)): labels[i] = 10
                elif (filename == labels_map.get(11)): labels[i] = 11
                elif (filename == labels_map.get(12)): labels[i] = 12
                elif (filename == labels_map.get(13)): labels[i] = 13
                elif (filename == labels_map.get(14)): labels[i] = 14
                elif (filename == labels_map.get(15)): labels[i] = 15
                elif (filename == labels_map.get(16)): labels[i] = 16
                elif (filename == labels_map.get(17)): labels[i] = 17
                elif (filename == labels_map.get(18)): labels[i] = 18
                elif (filename == labels_map.get(19)): labels[i] = 19
                elif (filename == labels_map.get(20)): labels[i] = 20
                elif (filename == labels_map.get(21)): labels[i] = 21
                elif (filename == labels_map.get(22)): labels[i] = 22
                elif (filename == labels_map.get(23)): labels[i] = 23
                elif (filename == labels_map.get(24)): labels[i] = 24
                elif (filename == labels_map.get(25)): labels[i] = 25
                elif (filename == labels_map.get(26)): labels[i] = 26
                elif (filename == labels_map.get(27)): labels[i] = 27
                elif (filename == labels_map.get(28)): labels[i] = 28
                          
                i+=1
    return (images, labels)

def get_data(batch_size : int):
    data = load_images(training_dataset_size)[0]
    training_images = data[0]
    training_labels = data[1]
    test_images = data[0]
    test_labels = data[1]
    
    train_dataset = tf.data.Dataset.from_tensor_slices((training_images, training_labels))
    test_dataset = tf.data.Dataset.from_tensor_slices((test_images, test_labels))

    train_dataset = train_dataset.map(lambda image, label: (float(image) / 255.0, label))
    test_dataset = test_dataset.map(lambda image, label: (float(image) / 255.0, label))

    train_dataset = train_dataset.batch(batch_size).shuffle(500)
    test_dataset = test_dataset.batch(batch_size).shuffle(500)

    return (train_dataset, test_dataset)

def training_phase():
    learning_rate = 0.1
    batch_size = 64
    epochs = 5

    (train_dataset, test_dataset) = get_data(batch_size)

    model = NeuralNetwork()

    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    optimizer = tf.keras.optimizers.SGD(learning_rate)
    metrics = ['accuracy']
    model.compile(optimizer, loss_fn, metrics)

    print('\nFitting:')
    model.fit(train_dataset, epochs=epochs)
        
    print('\nEvaluating:')
    (test_loss, test_accuracy) = model.evaluate(test_dataset)
    print(f'\nTest accuracy: {test_accuracy * 100:>0.1f}%, test loss: {test_loss:>8f}')

    model.save('outputs/model')

training_phase()
#def get_data(batch_size: int) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
