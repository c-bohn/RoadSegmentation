import tensorflow as tf
from config import config
import keras
from tensorflow.keras import layers


def u_net_modified(pixels_placeholder, is_training_placeholder):
    is_training = is_training_placeholder
    inputs = pixels_placeholder
    conv1 = tf.layers.conv2d(
        inputs=inputs,
        filters=64,
        kernel_size=[3, 3],
        padding="same",
        strides=1,
        activation=None,
        dilation_rate=config['dilation_rate']
    )
    if config['use_batch_norm']:
        conv1 = tf.layers.batch_normalization(conv1, training=is_training)
    conv1 = tf.nn.relu(conv1)
    conv1 = tf.layers.conv2d(
        inputs=conv1,
        filters=64,
        kernel_size=[3, 3],
        padding="same",
        strides=1,
        activation=None,
        dilation_rate=config['dilation_rate']
    )
    if config['use_batch_norm']:
        conv1 = tf.layers.batch_normalization(conv1, training=is_training)
    conv1 = tf.nn.relu(conv1)

    drop1 = tf.keras.layers.SpatialDropout2D(
        config['spatial_dropout_rate'],
        data_format=None
    )(conv1)

    pool1 = tf.layers.max_pooling2d(
        inputs=drop1,
        pool_size=2,
        strides=2,
        padding='same'
    )
    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=128,
        kernel_size=[3, 3],
        padding="same",
        strides=1,
        activation=None,
        dilation_rate=config['dilation_rate']
    )

    if config['use_batch_norm']:
        conv2 = tf.layers.batch_normalization(conv2, training=is_training)
    
    conv2 = tf.nn.relu(conv2)
    conv2 = tf.layers.conv2d(
        inputs=conv2,
        filters=128,
        kernel_size=[3, 3],
        padding="same",
        strides=1,
        activation=None,
        dilation_rate=config['dilation_rate']
    )

    if config['use_batch_norm']:
        conv2 = tf.layers.batch_normalization(conv2, training=is_training)

    conv2 = tf.nn.relu(conv2)
    drop2 = tf.keras.layers.SpatialDropout2D(
        config['spatial_dropout_rate'], 
        data_format=None
    )(conv2)

    pool2 = tf.layers.max_pooling2d(
        inputs=drop2,
        pool_size=2,
        strides=2,
        padding='same'
    )
    conv3 = tf.layers.conv2d(
        inputs=pool2,
        filters=256,
        kernel_size=[3, 3],
        padding="same",
        strides=1,
        activation=None,
        dilation_rate=config['dilation_rate']
    )
    if config['use_batch_norm']:
        conv3 = tf.layers.batch_normalization(conv3, training=is_training)
    conv3 = tf.nn.relu(conv3)
    conv3 = tf.layers.conv2d(
        inputs=conv3,
        filters=256,
        kernel_size=[3, 3],
        padding="same",
        strides=1,
        activation=None,
        dilation_rate=config['dilation_rate']
    )
    if config['use_batch_norm']:
        conv3 = tf.layers.batch_normalization(conv3, training=is_training)
    conv3 = tf.nn.relu(conv3)

    # drop3 = tf.layers.dropout(conv3, rate=config['dropout_rate'])
    drop3 = tf.keras.layers.SpatialDropout2D(
        config['spatial_dropout_rate'], 
        data_format=None
    )(conv3)

    pool3 = tf.layers.max_pooling2d(
        inputs=drop3,
        pool_size=2,
        strides=2,
        padding='same'
    )
    conv4 = tf.layers.conv2d(
        inputs=pool3,
        filters=512,
        kernel_size=[3, 3],
        padding="same",
        strides=1,
        activation=None,
        dilation_rate=config['dilation_rate']
    )
    if config['use_batch_norm']:
        conv4 = tf.layers.batch_normalization(conv4, training=is_training)
    conv4 = tf.nn.relu(conv4)
    conv4 = tf.layers.conv2d(
        inputs=conv4,
        filters=512,
        kernel_size=[3, 3],
        padding="same",
        strides=1,
        activation=None,
        dilation_rate=config['dilation_rate']
    )
    if config['use_batch_norm']:
        conv4 = tf.layers.batch_normalization(conv4, training=is_training)
    conv4 = tf.nn.relu(conv4)
    # drop4 = tf.layers.dropout(conv4, rate=0.5)
    drop4 = tf.keras.layers.SpatialDropout2D(
        config['spatial_dropout_rate'], data_format=None)(conv4)  # xx
    pool4 = tf.layers.max_pooling2d(
        inputs=drop4,
        pool_size=2,
        strides=2,
        padding='same'
    )
    conv5 = tf.layers.conv2d(
        inputs=pool4,
        filters=1024,
        kernel_size=[3, 3],
        padding="same",
        strides=1,
        activation=None,
        dilation_rate=config['dilation_rate']
    )
    if config['use_batch_norm']:
        conv5 = tf.layers.batch_normalization(conv5, training=is_training)
    conv5 = tf.nn.relu(conv5)
    conv5 = tf.layers.conv2d(
        inputs=conv5,
        filters=1024,
        kernel_size=[3, 3],
        padding="same",
        strides=1,
        activation=None,
        dilation_rate=config['dilation_rate']
    )
    if config['use_batch_norm']:
        conv5 = tf.layers.batch_normalization(conv5, training=is_training)
    conv5 = tf.nn.relu(conv5)
    # drop5 = tf.layers.dropout(conv5, rate=0.5)
    drop5 = tf.keras.layers.SpatialDropout2D(
        config['spatial_dropout_rate'], data_format=None)(conv5)  # xx
    up6 = tf.layers.conv2d_transpose(
        inputs=drop5,
        filters=512,
        kernel_size=[2, 2],
        padding="same",
        strides=2,
        activation=tf.nn.relu
    )
    merge6 = tf.concat([drop4, up6], axis=3)
    conv6 = tf.layers.conv2d(
        inputs=merge6,
        filters=512,
        kernel_size=[3, 3],
        padding="same",
        strides=1,
        activation=None,
        dilation_rate=config['dilation_rate']
    )
    if config['use_batch_norm']:
        conv6 = tf.layers.batch_normalization(conv6, training=is_training)
    conv6 = tf.nn.relu(conv6)
    conv6 = tf.layers.conv2d(
        inputs=conv6,
        filters=512,
        kernel_size=[3, 3],
        padding="same",
        strides=1,
        activation=None,
        dilation_rate=config['dilation_rate']
    )
    if config['use_batch_norm']:
        conv6 = tf.layers.batch_normalization(conv6, training=is_training)
    conv6 = tf.nn.relu(conv6)

    drop6 = tf.keras.layers.SpatialDropout2D(
        config['spatial_dropout_rate'],
        data_format=None
    )(conv6)  # xx

    up7 = tf.layers.conv2d_transpose(
        inputs=drop6,
        filters=256,
        kernel_size=[2, 2],
        padding="same",
        strides=2,
        activation=tf.nn.relu,
    )
    merge7 = tf.concat([conv3, up7], axis=3)
    conv7 = tf.layers.conv2d(
        inputs=merge7,
        filters=256,
        kernel_size=[3, 3],
        padding="same",
        strides=1,
        activation=None,
        dilation_rate=config['dilation_rate']
    )
    if config['use_batch_norm']:
        conv7 = tf.layers.batch_normalization(conv7, training=is_training)
    conv7 = tf.nn.relu(conv7)
    conv7 = tf.layers.conv2d(
        inputs=conv7,
        filters=256,
        kernel_size=[3, 3],
        padding="same",
        strides=1,
        activation=None,
        dilation_rate=config['dilation_rate']
    )
    if config['use_batch_norm']:
        conv7 = tf.layers.batch_normalization(conv7, training=is_training)
    conv7 = tf.nn.relu(conv7)

    # drop7 = tf.layers.dropout(conv7, rate=config['dropout_rate'])
    drop7 = tf.keras.layers.SpatialDropout2D(
        config['spatial_dropout_rate'],
         data_format=None
    )(conv7)  # xx

    up8 = tf.layers.conv2d_transpose(
        inputs=drop7,
        filters=128,
        kernel_size=[2, 2],
        padding="same",
        strides=2,
        activation=tf.nn.relu,
    )
    merge8 = tf.concat([conv2, up8], axis=3)
    conv8 = tf.layers.conv2d(
        inputs=merge8,
        filters=128,
        kernel_size=[3, 3],
        padding="same",
        strides=1,
        activation=None,
        dilation_rate=config['dilation_rate']
    )
    if config['use_batch_norm']:
        conv8 = tf.layers.batch_normalization(conv8, training=is_training)
    conv8 = tf.nn.relu(conv8)
    conv8 = tf.layers.conv2d(
        inputs=conv8,
        filters=128,
        kernel_size=[3, 3],
        padding="same",
        strides=1,
        activation=None,
        dilation_rate=config['dilation_rate']
    )
    if config['use_batch_norm']:
        conv8 = tf.layers.batch_normalization(conv8, training=is_training)
    conv8 = tf.nn.relu(conv8)

    drop8 = tf.keras.layers.SpatialDropout2D(
        config['spatial_dropout_rate'], 
        data_format=None
    )(conv8)  # xx

    up9 = tf.layers.conv2d_transpose(
        inputs=drop8,
        filters=64,
        kernel_size=[2, 2],
        padding="same",
        strides=2,
        activation=tf.nn.relu,
        )
    merge9 = tf.concat([conv1, up9], axis=3)
    conv9 = tf.layers.conv2d(
        inputs=merge9,
        filters=64,
        kernel_size=[3, 3],
        padding="same",
        strides=1,
        activation=None,
        dilation_rate=config['dilation_rate']
    )
    if config['use_batch_norm']:
        conv9 = tf.layers.batch_normalization(conv9, training=is_training)
    conv9 = tf.nn.relu(conv9)
    conv9 = tf.layers.conv2d(
        inputs=conv9,
        filters=64,
        kernel_size=[3, 3],
        padding="same",
        strides=1,
        activation=None,
        dilation_rate=config['dilation_rate']
    )
    if config['use_batch_norm']:
        conv9 = tf.layers.batch_normalization(conv9, training=is_training)
    conv9 = tf.nn.relu(conv9)

    drop9 = tf.keras.layers.SpatialDropout2D(
        config['spatial_dropout_rate'],
        data_format=None
    )(conv9)

    conv9 = tf.layers.conv2d(
        inputs=drop9,
        filters=2,
        kernel_size=[3, 3],
        padding="same",
        strides=1,
        activation=None,
        dilation_rate=config['dilation_rate']
    )

    logits_placeholder = conv9
    predictions_placeholder = 1 - tf.argmax(logits_placeholder, axis=3)

    return logits_placeholder, predictions_placeholder
