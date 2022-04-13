import sys
import shutil

import PIL.Image
import matplotlib.pyplot as plt
import numpy as np
import os
import PIL

import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

"""
Specs
our neural network generator, which will be called make nn.py. It will take two
command-line arguments: a directory in which the images are located, and the name
of the neural network file to create and save to disk. This program may assume that
all training image file names begin with c or d, for “cat” and “dog” respectively.
"""


def image_stats(filePath):
    """
    Takes in a file path and tries to get dimensions
    :param filePath: OS PATH
    :return: img_width, img_height
    """
    try:
        filename = os.listdir(filePath)[0]
        file = os.path.join(filePath, filename)
    except:
        print("Failure to get image")
        exit(1)
    try:
        img = PIL.Image.open(file)
        img_width = img.width
        img_height = img.height
    except:
        print("Error with loading image and finding width/height defaulting to 100x100")
        return (100, 100)
    return img_width, img_height


def create_labels(file_path):
    # TODO CHECK IF DIR Exists?
    try:
        dir_name = os.path.join(file_path, "dog")
        os.mkdir(dir_name)
    except:
        print("Error Making Directory")
    try:
        dir_name = os.path.join(file_path, "cat")
        os.mkdir(dir_name)
    except:
        print("Error Making Directory")
    # move files into there
    pass
    try:
        cat_dir = os.path.join(file_path, "cat")
        dog_dir = os.path.join(file_path, "dog")
        for filename in os.listdir(file_path):
            f = os.path.join(file_path, filename)
            if filename.__contains__('c'):
                shutil.move(f, cat_dir)
            elif filename.__contains__('d'):
                shutil.move(f, dog_dir)
    except:
        pass


if __name__ == '__main__':

    # Initialization vars
    # TODO Check If img =-1
    batch_size = 32
    img_height = 100
    img_width = 100
    epochs = 32

    # arguments should take in two command-line arguments Directory and Name of neural network file
    arguments = sys.argv

    if len(arguments) != 3:
        print(f"Found: {len(arguments) - 1} Expected: 2")
        exit(1)
    if os.path.isdir(arguments[1]):
        filePath = arguments[1]
    else:
        print(f"Path: {arguments[1]} Not Found")
        exit(1)

    # Check a gpu is being used
    # TODO
    # physical_devices = tf.config.list_physical_devices('GPU')
    # print(physical_devices)
    # This program may assume that all training image file names begin with c or d, for “cat” and “dog” respectively.

    # Prepare

    img_width_height = image_stats(filePath)

    # TODO See if adam wants this part but to move forward I believe we have to do this
    create_labels(filePath)

    # TODO issue with handling the labels i need to some how seperate the dogs and the cats
    train_ds = tf.keras.utils.image_dataset_from_directory(
        "/home/domenico/PycharmProjects/hw4/cats-and-dogs/",
        labels="inferred",
        seed=123,
        image_size=(img_width_height[1], img_width_height[0]),
        batch_size=batch_size)
    class_names = train_ds.class_names
    print(class_names)

    AUTOTUNE = tf.data.AUTOTUNE

    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)

    normalization_layer = layers.Rescaling(1. / 255)

    normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
    image_batch, labels_batch = next(iter(normalized_ds))
    first_image = image_batch[0]
    # Notice the pixel values are now in `[0,1]`.
    print(np.min(first_image), np.max(first_image))

    num_classes = len(class_names)

    # Used keras page on https://www.tensorflow.org/tutorials/keras/classification#make_predictions to make model
    # Also https://www.tensorflow.org/tutorials/images/classification
    # https://machinelearningmastery.com/how-to-accelerate-learning-of-deep-neural-networks-with-batch-normalization/
    # https://www.sicara.ai/blog/2019-10-31-convolutional-layer-convolution-kernel
    model = Sequential(
        [
            # https://keras.io/api/layers/normalization_layers/batch_normalization/
            # layers.Rescaling(1. / 255, input_shape=(img_height, img_width, 3)),

            layers.Input(shape=(img_height, img_width, 3)),
            layers.Conv2D(32, (3, 3), activation='relu'),
            layers.BatchNormalization(), # Makes each layer 0 to 1
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Dropout(.04),

            layers.Conv2D(64, (3,3), activation='relu'),
            layers.BatchNormalization(), # Makes each layer 0 to 1
            layers.MaxPooling2D(pool_size =(2,2)),
            layers.Dropout(.08),

            layers.Conv2D(128, (3,3), activation='relu'),
            layers.BatchNormalization(), # Makes each layer 0 to 1
            layers.MaxPooling2D(pool_size =(2,2)),
            layers.Dropout(.16),

            layers.Conv2D(256, (3,3), activation='relu'),
            layers.BatchNormalization(), # Makes each layer 0 to 1
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Dropout(.32),
            #Keeping dropout low because of https://machinelearningmastery.com/dropout-for-regularizing-deep-neural-networks/
            layers.Conv2D(512, (3, 3), activation='relu'),
            layers.BatchNormalization(),  # Makes each layer 0 to 1
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Dropout(.32),


            layers.Flatten(),
            # Seems more dense here the better or train for more epochs
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),

            # Dropout of .5 in order to ensure that the softmax determines correctly
            layers.Dropout(.8),
            layers.Dense(num_classes, activation='softmax'), # This goes to number of classes dense being 2 and activation is softmax therefore 0 or 1
            # Softmax is good for binary classification
        ]
    )

    #
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    # Summary of Arch for the model
    model.summary()

    #
    history = model.fit(
        train_ds,
        epochs=epochs
    )
    try:
        model.save(arguments[2] + '.dnn', include_optimizer=False, save_format='h5')
    except:
        print('Failed to Save Model')
