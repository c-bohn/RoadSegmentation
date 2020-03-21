"""training.py

This model trains a tensorflow model with a gradient descent
algorithm using the AdamOptimizer.

Early stopping is used to prevent overfitting. Currently we
use the U-net architecture but the architecture can it can easily
be replaced by changing the 'model' variable on line 75 by a custom
implementation.

All the hyperparamters for the model can be set in the config.py file
with according documentation in said file.
"""


import os
import random
import logging
import time

import numpy as np
from numpy.random import seed
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import set_random_seed
from keras import backend as K
from imgaug import augmenters as iaa

from model_architectures.u_net.u_net_modified import u_net_modified
from utilities.data_augmentation import apply_transforms

from config import (
    training_images_path,
    training_labels_path, 
    validation_images_path, 
    validation_labels_path,
    test_images_path, 
    predictions_path
)

# unique name for the log file with a timestamp:
timestr = time.strftime("%Y%m%d_%H%M%S")


def training(config):
    # Setting random seed
    SEED = 42
    seed(SEED)
    np.random.seed(SEED)
    tf.set_random_seed(SEED)
    tf.reset_default_graph()

    print(config)
    logging.info(config)

    # the data placeholders:
    pixels_placeholder = tf.placeholder(
        tf.float32,
        [None, None, None, 3],          # [batch_size, height, width, 3]
        name="pixels_placeholder"
    )
    labels_placeholder = tf.placeholder(
        tf.float32, 
        [None, None, None, 2],          # [batch_size, height, width, 2]
        name="labels_placeholder"
    )

    # whether to run batchnorm in training or inference mode
    is_training_placeholder = tf.placeholder(
        tf.bool, name="is_training_placeholder"
    )

    # our model:
    logits_placeholder, predictions_placeholder = u_net_modified(
        pixels_placeholder, is_training_placeholder
    )
    logits_placeholder = tf.identity(
        logits_placeholder, name="logits_placeholder"
    )
    predictions_placeholder = tf.identity(
        predictions_placeholder, name="predictions_placeholder"
    )

    # the number of parameters of the model:
    num_params = np.sum([
        np.product([xi.value for xi in x.get_shape()])
        for x in tf.all_variables()
    ])
    print('Number of parameters = {}'.format(num_params))
    logging.info('Number of parameters = {}'.format(num_params))


    # The class weights
    class_weights = tf.constant([[config['class_weight'], 1.]])
    # Deduce weights for batch samples based on their true label.
    weights = tf.reduce_sum(class_weights * labels_placeholder, axis=3)
    # Compute your (unweighted) softmax cross entropy loss.
    unweighted_losses = tf.nn.softmax_cross_entropy_with_logits_v2(
        labels=labels_placeholder,
        logits=logits_placeholder
    )
    # apply the weights, relying on broadcasting of the multiplication.
    weighted_losses = unweighted_losses * weights

    # reduce the result to get your final loss
    loss = tf.reduce_mean(weighted_losses)
    loss = tf.identity(loss, name='loss')

    # The accuracy of our predictions, this is what we really want to 
    # maximize, the loss function is merely a differentiable proxy for it.
    abs_diffs = np.abs(
        tf.cast(
            predictions_placeholder,
            tf.float32
        ) - labels_placeholder[:, :, :, 0])
    accuracy = 1 - tf.reduce_mean(abs_diffs)
    accuracy = tf.identity(accuracy, name="accuracy")

    # learning rate decay:
    global_step = tf.Variable(0, trainable=False)
    if config['use_exp_learning_rate_decay']:
        learning_rate = tf.train.exponential_decay(
            config['learning_rate'],
            global_step=global_step,
            decay_steps=800,
            decay_rate=0.97,
            staircase=False
        )
    else:
        learning_rate = config['learning_rate']

    # the optimizer:
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

    # the training operation
    train_op = optimizer.minimize(loss, global_step=global_step)

    # initializes the session
    init = tf.group(tf.global_variables_initializer(),
                    tf.local_variables_initializer())

    # the local_variables_initializer is needed for the internal variables
    #  created by tf.metrics.accuracy()
    session = tf.Session()
    session.run(init)
    K.set_session(session)

    # Saves the model if the validation performance improves; keeps the 
    # most recent 3 models
    saver = tf.train.Saver(max_to_keep=3, save_relative_paths=True)

    # Writes the loss values to disk, for visualization with Tensorboard
    summary_train_loss = tf.summary.scalar('train_loss', loss)
    summary_validation_loss = tf.summary.scalar('validation_loss', loss)
    train_summary_dir = 'results/train_summary'
    train_summary_writer = tf.summary.FileWriter(
        train_summary_dir, session.graph
    )
    valid_summary_dir = 'results/validation_summary'
    valid_summary_writer = tf.summary.FileWriter(
        valid_summary_dir, session.graph
    )

    # the np arrays holding the data:
    training_images = np.zeros([90, 400, 400, 3])
    training_labels = np.zeros([90, 400, 400, 2])
    validation_images = np.zeros([20, 400, 400, 3])
    validation_labels = np.zeros([20, 400, 400, 2])

    # load the training data:
    for i in range(90):
        images = os.listdir(training_images_path)
        # the filenames for the images and labels are the same, so to load
        #  them we just need to prepend the images- or labels-path
        filename = images[i]

        # load that image and its label:
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

        # the matrix entries don't span the entire range [0, 1] after 
        # they get loaded from the image, this fixes the min element 
        # to 0 and the max to 1:
        training_labels[i, :, :, 0] -= np.min(training_labels[i, :, :])
        training_labels[i, :, :, 0] /= np.max(training_labels[i, :, :])

    # we want one-hot encodings for the labels:
    training_labels[training_labels >= 0.5] = 1
    training_labels[training_labels < 0.5] = 0

    # P(not road) = 1 - P(road)
    training_labels[:, :, :, 1] = 1 - training_labels[:, :, :, 0]


    # load the validation data:
    for i in range(20):
        images = os.listdir(validation_images_path)
        filename = images[i]

        # load that image and its label:
        image = plt.imread(validation_images_path + filename)
        validation_images[i, :, :, :] = image[:, :, :3]

        label = plt.imread(validation_labels_path + filename)
        if len(label.shape) == 3:
            validation_labels[i, :, :, 0] = label[:, :, 0]
        else:
            validation_labels[i, :, :, 0] = label[:, :]

        # Again, we normalize the validation images to fit into [0, 1]:
        validation_labels[i, :, :, 0] -= np.min(validation_labels[i, :, :])
        validation_labels[i, :, :, 0] /= np.max(validation_labels[i, :, :])

    # we want one-hot encodings for the labels:
    validation_labels[validation_labels >= 0.5] = 1
    validation_labels[validation_labels < 0.5] = 0

    # P(not road) = 1 - P(road)
    validation_labels[:, :, :, 1] = 1 - validation_labels[:, :, :, 0]


    # train the model using early stopping to avoid overfitting:
    best_validation_accuracy = 0.
    epochs_since_last_improvement = 0
    train_step = 0
    val_step = 0
    for epoch in range(config['num_epochs']):
        logging.info("Starting epoch {}, current learning rate = {}".format(
            epoch+1, learning_rate.eval(session=session))
        )
        print("Starting epoch {}, current learning rate = {}".format(
            epoch+1, learning_rate.eval(session=session))
        )

        # training:
        is_training = True
        for batch in range(18):
            train_step += 1
            training_images_batch = (
                training_images[5 * batch: 5 * (batch + 1), :, :, :]
            )
            training_labels_batch = (
                training_labels[5 * batch: 5 * (batch + 1), :, :, :]
            )

            # Augmentation: randomly transform the batch
            (
                training_images_batch_aug, 
                training_labels_batch_aug
            ) = apply_transforms(
                training_images_batch,
                training_labels_batch[:, :, :, 0]
            )
            training_labels_batch_aug = np.stack(
                (training_labels_batch_aug, 1 - training_labels_batch_aug),
                axis=-1
            )
            print('Augmented training data for the batch')

            
            _, train_accuracy, train_loss, train_summary = session.run(
                [train_op, accuracy, loss, summary_train_loss],
                feed_dict={
                    pixels_placeholder: training_images_batch_aug,
                    labels_placeholder: training_labels_batch_aug,
                    is_training_placeholder: is_training
                }
            )
            train_summary_writer.add_summary(train_summary, train_step)
            logging.info(
                "Epoch {}, batch {}:\ttraining loss   = {}, accuracy = {}"
                .format(epoch + 1, batch + 1, train_loss, train_accuracy)
            )
            print(
                "Epoch {}, batch {}:\ttraining loss   = {}, accuracy = {}"
                .format(epoch + 1, batch + 1, train_loss, train_accuracy)
            )

        # validation
        is_training = False
        val_loss_sum = 0
        val_accuracy_sum = 0
        for batch in range(4):
            val_step += 1
            validation_images_batch = (
                validation_images[5 * batch: 5 * (batch + 1), :, :, :]
            )
            validation_labels_batch = (
                validation_labels[5 * batch: 5 * (batch + 1), :, :, :]
            )

            val_accuracy_batch, val_loss_batch, val_summary = session.run(
                [accuracy, loss, summary_validation_loss],
                feed_dict={
                    pixels_placeholder: validation_images_batch,
                    labels_placeholder: validation_labels_batch,
                    is_training_placeholder: is_training
                }
            )
            valid_summary_writer.add_summary(val_summary, val_step)
            val_loss_sum += val_loss_batch
            val_accuracy_sum += val_accuracy_batch
        val_accuracy = val_accuracy_sum / 4
        logging.info("\t\t\tvalidation loss = {}, accuracy = {}\n".format(
            val_loss_sum / 4, val_accuracy))
        print("\t\t\tvalidation loss = {}, accuracy = {}\n".format(
            val_loss_sum / 4, val_accuracy))

        # early stopping:
        if val_accuracy > best_validation_accuracy:
            epochs_since_last_improvement = 0
            best_validation_accuracy = val_accuracy
            config['best_model'] = epoch + 1
            checkpt_path = saver.save(session, os.path.join(
                config['model_dir'], timestr), global_step=epoch + 1)
            logging.info("Model saved in file: %s" % checkpt_path)
            print("Model saved in file: %s" % checkpt_path)
        else:
            epochs_since_last_improvement += 1
        if epochs_since_last_improvement > config['patience']:
            logging.info(
                'No improvement in validation performance for {} epochs, stopping training now'
                .format(config['patience'])
            )
            logging.info(
                'Validation accuracy of optimal model #{}: {}'
                .format(epoch - config['patience'], best_validation_accuracy)
            )
            print('No improvement in validation performance for {} epochs, stopping training now'
                  .format(config['patience'])
            )
            print('Validation accuracy of optimal model #{}: {}'
                .format(epoch - config['patience'], best_validation_accuracy)
            )
            break

    if not config['predict_after_training']:
        return



    # predict on the test data:
    logging.info('\n\nDone training, predicting on the test data now...')
    print('\n\nDone training, predicting on the test data now...')

    # Make sure the path has a traililing slash.
    load_path = os.path.join(predictions_path, '')
    # Create directory if it doesn't exist yet.
    if not os.path.exists(load_path):
        print("Creating directory {}".format(load_path))
        os.makedirs(load_path)


    test_images = np.zeros([94, 608, 608, 3])
    images = os.listdir(test_images_path)
    for i in range(94):
        filename = images[i]
        test_images[i, :, :, :] = plt.imread(test_images_path + filename)[:, :, :3]
        test_images[i, :, :, :] = (
            test_images[i, :, :, :] - np.mean(test_images[i, :, :, :])
        )

    print('loaded all test data')

    for i in range(94):
        _, predictions_test = session.run(
            [logits_placeholder, predictions_placeholder],
            feed_dict={
                pixels_placeholder: np.expand_dims(test_images[i, :, :, :], 0),
                is_training_placeholder: False
            }
        )
        filename = images[i]
        plt.imsave(
            str(predictions_path) + 'prediction_' + filename,
            predictions_test.squeeze(), cmap='gray'
        )

    logging.info('All done')
    print('All done')


if __name__ == '__main__':
    from config import config

    logging.basicConfig(filename='logs_' + timestr + '.log',
                        level=logging.INFO, filemode='w')
    training(config)
