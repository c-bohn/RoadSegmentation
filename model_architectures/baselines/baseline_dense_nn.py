import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

if __file__:
    file_path = os.path.realpath(__file__)
    ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(file_path)))
    print("Accessing root modules...")
    sys.path.append(ROOT_DIR)
    sys.path.append(os.path.join(ROOT_DIR, 'utilities'))
else:
    print(
        "File run in interactive environment.",
        "Cannot append ROOT_DIR to sys.path..."
    )

from config import (
    training_images_path,
    training_labels_path,
    test_images_path
)

def baseline_dense_nn():
    patches_placeholder = tf.placeholder(tf.float32, [None, 16, 16, 3], name="training_patch_placeholder")
    patch_labels_placeholder = tf.placeholder(tf.float32, [None, 1], name='training_label_placeholder')

    def model(training_patches_placeholder):
        input_layer = training_patches_placeholder
        input_flat = tf.contrib.layers.flatten(input_layer)
        hidden_layer_1 = tf.layers.dense(inputs=input_flat, units=256, activation=tf.nn.relu)
        hidden_layer_2 = tf.layers.dense(inputs=hidden_layer_1, units=16, activation=tf.nn.relu)
        output_layer = tf.layers.dense(inputs=hidden_layer_2, units=1, activation=None)
        return output_layer

    predictions_placeholder = model(patches_placeholder)
    predictions_placeholder = tf.identity(predictions_placeholder, name="training_predictions_placeholder")

    loss = tf.losses.sigmoid_cross_entropy(multi_class_labels=patch_labels_placeholder,
                                           logits=predictions_placeholder)
    loss = tf.identity(loss, name='loss')

    optimizer = tf.train.AdamOptimizer(learning_rate=1e-5)
    train_op = optimizer.minimize(loss)

    init = tf.global_variables_initializer()

    session = tf.Session()
    session.run(init)

    training_images = np.zeros([90, 400, 400, 3])
    training_labels = np.zeros([90, 400, 400, 1])

    for i in range(90):
        images = os.listdir(training_images_path)
        filename = images[i]
        image = plt.imread(training_images_path + filename)
        training_images[i, :, :, :] = image[:, :, :3]

        label = plt.imread(training_labels_path + filename)
        if i == 0:
            print("Label shape: ")
            print(label.shape)
            print(len(label.shape))
        if len(label.shape) == 3:
            training_labels[i, :, :, 0] = label[:, :, 0]
        else:
            training_labels[i, :, :, 0] = label[:, :]

        training_labels[i, :, :, 0] -= np.min(training_labels[i, :, :])
        training_labels[i, :, :, 0] /= np.max(training_labels[i, :, :])

    training_labels[training_labels >= 0.5] = 1
    training_labels[training_labels < 0.5] = 0

    training_patches = np.zeros((25*25, 16, 16, 3))
    training_patch_labels = np.zeros((25*25, 1))
    for epoch in range(10):
        for image in range(90):
            for j in range(25):
                for i in range(25):
                    training_patches[j*25+i, :, :, :] = training_images[image, i*16:(i+1)*16, j*16:(j+1)*16, :]
                    if np.mean(training_labels[image, i*16:(i+1)*16, j*16:(j+1)*16, :]) >= 0.7:
                        training_patch_labels[j*25+i, 0] = 1
                    else:
                        training_patch_labels[i*25+j, 0] = 0
            _, train_loss = session.run([train_op, loss],
                                        feed_dict={patches_placeholder: training_patches,
                                                   patch_labels_placeholder: training_patch_labels})
            print('Epoch {}, image {}: training loss = {}'.format(epoch, image, train_loss))

    print('\n\nTraining done\nPredicting now...')

    test_images = np.zeros((94, 608, 608, 3))
    for i in range(90):
        images = os.listdir(test_images_path)
        filename = images[i]
        image = plt.imread(test_images_path + filename)
        test_images[i, :, :, :] = image[:, :, :3]

    test_patches = np.zeros((38*38, 16, 16, 3))
    test_patch_labels = np.zeros((94, 38*38, 1))
    for image in range(94):
        for j in range(38):
            for i in range(38):
                test_patches[j * 38 + i, :, :, :] = test_images[image, i * 16:(i + 1) * 16, j * 16:(j + 1) * 16, :]
        test_patch_labels[image, :, :] = session.run(predictions_placeholder, feed_dict={patches_placeholder: test_patches})
        print('Test image {} done'.format(image))

    # TODO: predict on val set, compute accuracy score from that prediction

if __name__ == '__main__':
    baseline_dense_nn()
