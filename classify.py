import sys

import numpy as np
import tensorflow as tf
from PIL import Image

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
        # This is potentially unsafe given the fact that dnn are executable code
        # Caution: TensorFlow models are code and it is important to be careful with untrusted code. See Using TensorFlow Securely for details.
        # https://github.com/tensorflow/tensorflow/blob/master/SECURITY.md
        # https://www.tensorflow.org/guide/keras/save_and_serialize
        loaded_model = tf.keras.models.load_model(arguments[1])
    except:
        print('Failed to load model')
        exit(1)

    class_labels = ['Cat', 'Dog']
    # Since first two arguments of arguments to get length of files we subtract total size - 2
    try:
        for image_idx in range(2, len(arguments)):
            # Ensure Images are of width 100 and size 100 by forcing it
            new_width = 100
            new_height = 100
            img = tf.keras.utils.load_img(arguments[image_idx])
            img = img.resize((new_width, new_height), Image.ANTIALIAS)
            img.save(arguments[image_idx])
            # All image_idx are images as given by specs
            img = tf.keras.utils.load_img(arguments[image_idx])
            img_array = tf.keras.utils.img_to_array(img)
            img_array = tf.expand_dims(img_array, 0)
            # predict model https://www.tensorflow.org/api_docs/python/tf/keras/Model#predict
            predictions = loaded_model.predict(img_array)
            score = tf.nn.softmax(predictions[0])

            print(
                "This image {} most likely belongs to {} with a {:.2f} percent confidence."
                    .format(arguments[image_idx], class_labels[np.argmax(score)], 100 * np.max(score))
            )
    except:
        print('Failure In args one or more arguments: {}'.format(arguments))
        exit(1)
