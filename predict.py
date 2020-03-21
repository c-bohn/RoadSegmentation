import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from imgaug import augmenters as iaa
from config import config, test_images_path, predictions_path


# Load the timestamp to specify which model to load.
timestamp = config['model_timestamp']

def predict(config):
    """Loads the model specified in the config file, and predicts the 94 provided
    test images. The paths for loading the test images and storing the predictions
    have to be specified in the config file.
    """

    # Make sure the path has a traililing slash.
    load_path = os.path.join(predictions_path, '')
    # Create directory if it doesn't exist yet.
    if not os.path.exists(load_path):
        print("Creating directory {}".format(load_path))
        os.makedirs(load_path)

    tf.reset_default_graph()

    session = tf.Session()
    saver = tf.train.import_meta_graph("{}{}-{}.meta".format(
        str(config['model_dir']),
        timestamp,
        str(config['best_model'])
    ))
    model_path = "{}{}-{}".format(
        str(config['model_dir']),
        timestamp,
        str(config['best_model'])
    )
    saver.restore(session, model_path)

    print("Restored {}model-{}-{}".format(
        str(config['model_dir']),
        timestamp,
        str(config['best_model'])
    ))

    test_images = np.zeros([config['num_test_imgs'], 608, 608, 3])
    images = os.listdir(test_images_path)
    for i in range(config['num_test_imgs']):
        filename = images[i]
        test_images[i, :, :, :] = plt.imread(test_images_path + filename)[:, :, :3]


    print('Loaded all test data')


    graph = tf.get_default_graph()
    pixels_placeholder = graph.get_tensor_by_name("pixels_placeholder:0")
    is_training_placeholder = graph.get_tensor_by_name("is_training_placeholder:0")
    logits_placeholder = graph.get_tensor_by_name("logits_placeholder:0")
    predictions_placeholder = graph.get_tensor_by_name("predictions_placeholder:0")


    for i in range(config['num_test_imgs']):
        logits_test, predictions_test = session.run(
            fetches=[logits_placeholder, predictions_placeholder],
            feed_dict={
                pixels_placeholder: np.expand_dims(test_images[i, :, :, :], 0),
                is_training_placeholder: False
            }
        )
        if config['postprocessing'] == True:
            # Erosion and dilation to remove small patches and connect larger close ones
            predictions_test *= 255
            predictions_test = predictions_test.astype('uint8')
            # a larger kernel would remove too much:
            kernel1 = np.ones((7, 7), np.uint8)
            predictions_test[0, :, :] = cv2.morphologyEx(
                predictions_test[0, :, :], 
                cv2.MORPH_OPEN, kernel1, 
                iterations=1
            )
            kernel2 = np.ones((15, 15), np.uint8) # maybe even larger?
            predictions_test[0, :, :] = cv2.morphologyEx(
                predictions_test[0, :, :], 
                cv2.MORPH_CLOSE, 
                kernel2, 
                iterations=1
            )
            predictions_test = predictions_test.astype(float)
            predictions_test /= 255

        probabilities = (tf.nn.softmax(logits_test)).eval(session=session)
        filename = images[i]

        plt.imsave(
            "{}prediction_{}".format(str(load_path), filename),
            predictions_test[0,:,:],
            cmap='gray'
        )
        print("{}/{}: Prediction for {} done.".format(
            str(i + 1),
            str(config['num_test_imgs']),
            filename
        ))
    print('All done')


if __name__ == '__main__':
    from config import config
    predict(config)
