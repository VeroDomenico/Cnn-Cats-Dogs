import sys

import matplotlib.pyplot as plt
import numpy
import numpy as np
import os
import PIL
import tensorflow as tf
from PIL import  Image
import cv2

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

"""
Specs:
A classifier, called classify.py. Its first argument is the neural network to load up.
All remaining arguments are image files to classify. It should print out one line for
each file, with its name and whether it contains a cat or a dog. It may not make any
assumptions about file names
"""

if __name__ == '__main__':
    # arguments should take in two command-line arguments Directory and Name of neural network file
    arguments = sys.argv
    try:
        loaded_model = tf.keras.models.load_model(arguments[1])
    except:
        print('Failed to load model')

    class_labels = ['Cat', 'Dog']
    # Since first two arguments of arguments to get length of files we subtract total size - 2
    for image_idx in range(2, len(arguments)):
        # Ensure Images are of width 100 and size 100
        new_width = 100
        new_height = 100
        img = tf.keras.utils.load_img(arguments[image_idx])
        img = img.resize((new_width, new_height), Image.ANTIALIAS)
        img.save(arguments[image_idx])
        #All image_idx are images as given by specs


        img = tf.keras.utils.load_img(arguments[image_idx])
        img_array = tf.keras.utils.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)

        predictions = loaded_model.predict(img_array)

        predictions = loaded_model.predict(img_array)
        score = tf.nn.softmax(predictions[0])

        print(
            "This image {} most likely belongs to {} with a {:.2f} percent confidence."
                .format(arguments[image_idx],class_labels[np.argmax(score)], 100 * np.max(score))
        )
    
