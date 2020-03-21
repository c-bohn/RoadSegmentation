"""Modification of the original 'predict.py' script to make it work
for the validation images (which are of different shape than the 
test images.) Was only used to generate images for the report but
not for the submissions on kaggle.
"""

import access_root_modules      # IMPORTANT: This import has to stay on top

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from imgaug import augmenters as iaa
from config import config, validation_images_path

timestamp = config['model_timestamp']

def predict_on_val(config):
    tf.reset_default_graph()

    session = tf.Session()
    saver = tf.train.import_meta_graph(str(config['model_dir']) + timestamp + '-' + str(config['best_model']) + '.meta')
    model_path = str(config['model_dir']) + timestamp + '-' + str(config['best_model'])
    saver.restore(session, model_path)

    print('Restored ' + str(config['model_dir'])  + 'model-' + str(config['best_model']))


    val_images = np.zeros([20, 400, 400, 3])
    images = os.listdir(validation_images_path)
    for i in range(20):
        filename = images[i]
        val_images[i, :, :, :] = plt.imread(validation_images_path + filename)[:, :, :3]


    print('Loaded all test data')


    graph = tf.get_default_graph()
    pixels_placeholder = graph.get_tensor_by_name("pixels_placeholder:0")
    is_training_placeholder = graph.get_tensor_by_name("is_training_placeholder:0")
    logits_placeholder = graph.get_tensor_by_name("logits_placeholder:0")
    predictions_placeholder = graph.get_tensor_by_name("predictions_placeholder:0")


    for i in range(20):
        logits_test, predictions_test = session.run([logits_placeholder, predictions_placeholder],
                                                    feed_dict={pixels_placeholder: np.expand_dims(val_images[i, :, :, :], 0),
                                                               is_training_placeholder: False})
        if config['postprocessing'] == True:
            # Erosion and dilation to remove small patches and connect larger close ones
            predictions_test *= 255
            predictions_test = predictions_test.astype('uint8')
            kernel1 = np.ones((7, 7), np.uint8) # a larger kernel would remove too much
            predictions_test[0, :, :] = cv2.morphologyEx(predictions_test[0, :, :], cv2.MORPH_OPEN, kernel1, iterations=1)
            kernel2 = np.ones((15, 15), np.uint8) # maybe even larger?
            predictions_test[0, :, :] = cv2.morphologyEx(predictions_test[0, :, :], cv2.MORPH_CLOSE, kernel2, iterations=1)
            predictions_test = predictions_test.astype(float)
            predictions_test /= 255

        probabilities = (tf.nn.softmax(logits_test)).eval(session=session)
        filename = images[i]
        plt.imsave('results/predictions_on_val_set/prediction_' + filename, probabilities[0,:,:,0], cmap='gray')
        print(str(i+1) + '/' + str(20) + ': Prediction for ' + filename + ' done')

    print('All done')


if __name__ == '__main__':
    from config import config
    predict_on_val(config)
