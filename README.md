# Homework-4-Convoluting-Cats-Dogs

Purpose:
Keep track of changes when finding correct neural network model through fine tuning.
Also try different arch 


100x100 commands for custom dataset 
for file in *.jpg; do convert $file -resize 100x100! $file; done


Classify class 

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


This image IMG_7191.JPG most likely belongs to Cat with a 71.22 percent confidence.
prediction [[1.3322309e-04 9.9986672e-01]]
score tf.Tensor([0.26899382 0.73100615], shape=(2,), dtype=float32)
This image cat?.jpg most likely belongs to Dog with a 73.10 percent confidence.

softmax effects prediction resulting in a 2d tensor  ([0.26899382 0.73100615])

This args max takes the max and returns a index printing the value and np.max(score) gives max which is locked between 0 and 1 resutling in the max decimal * 100 to result in percentage value
