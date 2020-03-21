import tensorflow as tf
from config import config
import keras
from tensorflow.keras import layers


def u_net_modified_deep(pixels_placeholder, is_training_placeholder):
    # indicates whether to run batchnorm in training or inference mode
    is_training = is_training_placeholder
    inputs = pixels_placeholder
    # conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
    conv1 = tf.layers.conv2d(inputs=inputs,
                             filters=64,
                             kernel_size=[3, 3],
                             padding="same",
                             strides=1,
                             activation=None)
    if config['use_batch_norm']:
        conv1 = tf.layers.batch_normalization(conv1, training=is_training)
    conv1 = tf.nn.relu(conv1)
    # conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
    conv1 = tf.layers.conv2d(inputs=conv1,
                             filters=64,
                             kernel_size=[3, 3],
                             padding="same",
                             strides=1,
                             activation=None)
    if config['use_batch_norm']:
        conv1 = tf.layers.batch_normalization(conv1, training=is_training)
    conv1 = tf.nn.relu(conv1)

    drop1 = tf.layers.dropout(conv1, rate=config['dropout_rate'])

    # pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    pool1 = tf.layers.max_pooling2d(inputs=drop1,
                                    pool_size=2,
                                    strides=2,
                                    padding='same')
    # conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
    conv2 = tf.layers.conv2d(inputs=pool1,
                             filters=128,
                             kernel_size=[3, 3],
                             padding="same",
                             strides=1,
                             activation=None)
    if config['use_batch_norm']:
        conv2 = tf.layers.batch_normalization(conv2, training=is_training)
    conv2 = tf.nn.relu(conv2)
    # conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
    conv2 = tf.layers.conv2d(inputs=conv2,
                             filters=128,
                             kernel_size=[3, 3],
                             padding="same",
                             strides=1,
                             activation=None)
    if config['use_batch_norm']:
        conv2 = tf.layers.batch_normalization(conv2, training=is_training)
    conv2 = tf.nn.relu(conv2)

    drop2 = tf.layers.dropout(conv2, rate=config['dropout_rate'])

    # pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    pool2 = tf.layers.max_pooling2d(inputs=drop2,
                                    pool_size=2,
                                    strides=2,
                                    padding='same')
    # conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
    conv3 = tf.layers.conv2d(inputs=pool2,
                             filters=256,
                             kernel_size=[3, 3],
                             padding="same",
                             strides=1,
                             activation=None)
    if config['use_batch_norm']:
        conv3 = tf.layers.batch_normalization(conv3, training=is_training)
    conv3 = tf.nn.relu(conv3)
    # conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
    conv3 = tf.layers.conv2d(inputs=conv3,
                             filters=256,
                             kernel_size=[3, 3],
                             padding="same",
                             strides=1,
                             activation=None)
    if config['use_batch_norm']:
        conv3 = tf.layers.batch_normalization(conv3, training=is_training)
    conv3 = tf.nn.relu(conv3)

    drop3 = tf.layers.dropout(conv3, rate=config['dropout_rate'])

    # pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    pool3 = tf.layers.max_pooling2d(inputs=drop3,
                                    pool_size=2,
                                    strides=2,
                                    padding='same')
    # conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
    conv4 = tf.layers.conv2d(inputs=pool3,
                             filters=512,
                             kernel_size=[3, 3],
                             padding="same",
                             strides=1,
                             activation=None)
    if config['use_batch_norm']:
        conv4 = tf.layers.batch_normalization(conv4, training=is_training)
    conv4 = tf.nn.relu(conv4)
    # conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
    conv4 = tf.layers.conv2d(inputs=conv4,
                             filters=512,
                             kernel_size=[3, 3],
                             padding="same",
                             strides=1,
                             activation=None)
    if config['use_batch_norm']:
        conv4 = tf.layers.batch_normalization(conv4, training=is_training)
    conv4 = tf.nn.relu(conv4)
    # drop4 = Dropout(0.5)(conv4)
    drop4 = tf.layers.dropout(conv4, rate=0.5)
    # pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)
    pool4 = tf.layers.max_pooling2d(inputs=drop4,
                                    pool_size=2,
                                    strides=2,
                                    padding='same')

    # conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
    conv4_2 = tf.layers.conv2d(inputs=pool4,
                               filters=1024,
                               kernel_size=[3, 3],
                               padding="same",
                               strides=1,
                               activation=None)
    if config['use_batch_norm']:
        conv4_2 = tf.layers.batch_normalization(conv4_2, training=is_training)
    conv4_2 = tf.nn.relu(conv4_2)
    # conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
    conv4_2 = tf.layers.conv2d(inputs=conv4_2,
                               filters=1024,
                               kernel_size=[3, 3],
                               padding="same",
                               strides=1,
                               activation=None)
    if config['use_batch_norm']:
        conv4_2 = tf.layers.batch_normalization(conv4_2, training=is_training)
    conv4_2 = tf.nn.relu(conv4_2)
    # drop4 = Dropout(0.5)(conv4)
    drop4_2 = tf.layers.dropout(conv4_2, rate=0.5)
    # pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)
    pool4_2 = tf.layers.max_pooling2d(inputs=drop4_2,
                                      pool_size=2,
                                      strides=2,
                                      padding='same')

    # conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
    conv5 = tf.layers.conv2d(inputs=pool4_2,
                             filters=2048,
                             kernel_size=[3, 3],
                             padding="same",
                             strides=1,
                             activation=None)
    if config['use_batch_norm']:
        conv5 = tf.layers.batch_normalization(conv5, training=is_training)
    conv5 = tf.nn.relu(conv5)
    # conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
    conv5 = tf.layers.conv2d(inputs=conv5,
                             filters=2048,
                             kernel_size=[3, 3],
                             padding="same",
                             strides=1,
                             activation=None)
    if config['use_batch_norm']:
        conv5 = tf.layers.batch_normalization(conv5, training=is_training)
    conv5 = tf.nn.relu(conv5)
    # drop5 = Dropout(0.5)(conv5)
    drop5 = tf.layers.dropout(conv5, rate=0.5)
    # up6 = Conv2D(512, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
    #     UpSampling2D(size=(2, 2))(drop5))
    up6_2 = tf.layers.conv2d_transpose(inputs=drop5,  # this is not exactly the same as the keras statement above
                                       filters=1024,
                                       kernel_size=[2, 2],
                                       padding="same",
                                       strides=2,
                                       activation=tf.nn.relu)
    # merge6 = concatenate([drop4, up6], axis=3)
    merge6_2 = tf.concat([drop4_2, up6_2], axis=3)
    # conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
    conv6_2 = tf.layers.conv2d(inputs=merge6_2,
                               filters=1024,
                               kernel_size=[3, 3],
                               padding="same",
                               strides=1,
                               activation=None)
    if config['use_batch_norm']:
        conv6_2 = tf.layers.batch_normalization(conv6_2, training=is_training)
    conv6_2 = tf.nn.relu(conv6_2)
    # conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)
    conv6_2 = tf.layers.conv2d(inputs=conv6_2,
                               filters=512,
                               kernel_size=[3, 3],
                               padding="same",
                               strides=1,
                               activation=None)
    if config['use_batch_norm']:
        conv6_2 = tf.layers.batch_normalization(conv6_2, training=is_training)
    conv6_2 = tf.nn.relu(conv6_2)

    drop6_2 = tf.layers.dropout(conv6_2, rate=config['dropout_rate'])

    # up6 = Conv2D(512, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
    #     UpSampling2D(size=(2, 2))(drop5))
    up6 = tf.layers.conv2d_transpose(inputs=drop6_2,  # this is not exactly the same as the keras statement above
                                     filters=512,
                                     kernel_size=[2, 2],
                                     padding="same",
                                     strides=2,
                                     activation=tf.nn.relu)
    # merge6 = concatenate([drop4, up6], axis=3)
    merge6 = tf.concat([drop4, up6], axis=3)
    # conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
    conv6 = tf.layers.conv2d(inputs=merge6,
                             filters=512,
                             kernel_size=[3, 3],
                             padding="same",
                             strides=1,
                             activation=None)
    if config['use_batch_norm']:
        conv6 = tf.layers.batch_normalization(conv6, training=is_training)
    conv6 = tf.nn.relu(conv6)
    # conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)
    conv6 = tf.layers.conv2d(inputs=conv6,
                             filters=512,
                             kernel_size=[3, 3],
                             padding="same",
                             strides=1,
                             activation=None)
    if config['use_batch_norm']:
        conv6 = tf.layers.batch_normalization(conv6, training=is_training)
    conv6 = tf.nn.relu(conv6)

    drop6 = tf.layers.dropout(conv6, rate=config['dropout_rate'])

    # up7 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
    #     UpSampling2D(size=(2, 2))(conv6))
    up7 = tf.layers.conv2d_transpose(inputs=drop6,  # this is not exactly the same as the keras statement above
                                     filters=256,
                                     kernel_size=[2, 2],
                                     padding="same",
                                     strides=2,
                                     activation=tf.nn.relu)
    # merge7 = concatenate([conv3, up7], axis=3)
    merge7 = tf.concat([conv3, up7], axis=3)
    # conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
    conv7 = tf.layers.conv2d(inputs=merge7,
                             filters=256,
                             kernel_size=[3, 3],
                             padding="same",
                             strides=1,
                             activation=None)
    if config['use_batch_norm']:
        conv7 = tf.layers.batch_normalization(conv7, training=is_training)
    conv7 = tf.nn.relu(conv7)
    # conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)
    conv7 = tf.layers.conv2d(inputs=conv7,
                             filters=256,
                             kernel_size=[3, 3],
                             padding="same",
                             strides=1,
                             activation=None)
    if config['use_batch_norm']:
        conv7 = tf.layers.batch_normalization(conv7, training=is_training)
    conv7 = tf.nn.relu(conv7)

    drop7 = tf.layers.dropout(conv7, rate=config['dropout_rate'])

    # up8 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
    #     UpSampling2D(size=(2, 2))(conv7))
    up8 = tf.layers.conv2d_transpose(inputs=drop7,  # this is not exactly the same as the keras statement above
                                     filters=128,
                                     kernel_size=[2, 2],
                                     padding="same",
                                     strides=2,
                                     activation=tf.nn.relu)
    # merge8 = concatenate([conv2, up8], axis=3)
    merge8 = tf.concat([conv2, up8], axis=3)
    # conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
    conv8 = tf.layers.conv2d(inputs=merge8,
                             filters=128,
                             kernel_size=[3, 3],
                             padding="same",
                             strides=1,
                             activation=None)
    if config['use_batch_norm']:
        conv8 = tf.layers.batch_normalization(conv8, training=is_training)
    conv8 = tf.nn.relu(conv8)
    # conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)
    conv8 = tf.layers.conv2d(inputs=conv8,
                             filters=128,
                             kernel_size=[3, 3],
                             padding="same",
                             strides=1,
                             activation=None)
    if config['use_batch_norm']:
        conv8 = tf.layers.batch_normalization(conv8, training=is_training)
    conv8 = tf.nn.relu(conv8)

    drop8 = tf.layers.dropout(conv8, rate=config['dropout_rate'])

    # up9 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
    #     UpSampling2D(size=(2, 2))(conv8))
    up9 = tf.layers.conv2d_transpose(inputs=drop8,  # this is not exactly the same as the keras statement above
                                     filters=64,
                                     kernel_size=[2, 2],
                                     padding="same",
                                     strides=2,
                                     activation=tf.nn.relu)
    # merge9 = concatenate([conv1, up9], axis=3)
    merge9 = tf.concat([conv1, up9], axis=3)
    # conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
    conv9 = tf.layers.conv2d(inputs=merge9,
                             filters=64,
                             kernel_size=[3, 3],
                             padding="same",
                             strides=1,
                             activation=None)
    if config['use_batch_norm']:
        conv9 = tf.layers.batch_normalization(conv9, training=is_training)
    conv9 = tf.nn.relu(conv9)
    # conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    conv9 = tf.layers.conv2d(inputs=conv9,
                             filters=64,
                             kernel_size=[3, 3],
                             padding="same",
                             strides=1,
                             activation=None)
    if config['use_batch_norm']:
        conv9 = tf.layers.batch_normalization(conv9, training=is_training)
    conv9 = tf.nn.relu(conv9)

    drop9 = tf.layers.dropout(conv9, rate=config['dropout_rate'])

    # conv9 = Conv2D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    conv9 = tf.layers.conv2d(inputs=drop9,
                             filters=2,
                             kernel_size=[3, 3],
                             padding="same",
                             strides=1,
                             activation=None)
    # conv10 = Conv2D(1, 1, activation='sigmoid')(conv9)
    # conv10 = tf.layers.conv2d(inputs=conv9,
    #                          filters=1,
    #                          kernel_size=[1, 1],
    #                          padding="same",
    #                          strides=1,
    #                          activation=tf.nn.sigmoid)
    # model = Model(input=inputs, output=conv10)
    #
    # model.compile(optimizer=Adam(lr=1e-4), loss='binary_crossentropy', metrics=['accuracy'])
    #
    # # model.summary()
    #
    # if (pretrained_weights):
    #     model.load_weights(pretrained_weights)
    #
    # return model

    logits_placeholder = conv9
    predictions_placeholder = 1 - tf.argmax(logits_placeholder, axis=3)

    return logits_placeholder, predictions_placeholder
