"""Run this script to create all the whitened images
in the data/whitened directory
"""


import numpy as np
import matplotlib.pyplot as plt
import os
from data_paths import (
    training_images_path,
    training_labels_path,
    validation_images_path,
    validation_labels_path,
    test_images,
    w_training_images_path,
    w_training_labels_path,
    w_validation_images_path,
    w_validation_labels_path,
    w_test_images
)

def create_whitened_labels(source_path, dest_path):
    images = load_images(source_path)
    save_images(images, dest_path)


def create_whitened_images(source_path, dest_path, debug=False):
    print("Whitening images from {} and store them at {}."
        .format(source_path, dest_path))
    images = load_images(source_path)
    X = flatten(images)
    X_whitened = zca(X)
    images_whitened = unflatten(X_whitened, images[0].shape)
    if debug:
        plt.imshow(images_whitened[1])
        plt.show()
    save_images(images_whitened, dest_path)


def create_whitened_test_images(source_path, dest_path):
    images = load_images(source_path)
    X = flatten(images)
    X_whitened = zca(X)
    images_whitened = unflatten(X_whitened, images[0].shape)
    filenames = os.listdir(source_path)
    save_images_with_names(images_whitened, dest_path, filenames)


def load_images(path):
    filenames = os.listdir(path)
    if filenames:
        filename = os.path.join(path, filenames[0])
        img_shape = plt.imread(filename).shape
    images = np.zeros([len(filenames), *img_shape])
    for i in range(len(images)):
        # the filenames for the images and labels are the same, so to load them we just need to prepend the images- or labels-path
        filename = os.path.join(path, filenames[i])

        # load that image and its label:
        image = plt.imread(filename)
        images[i] = image
    return  images


def save_images_with_names(images, path, filenames, normalize_img=True):
    if not os.path.exists(path):
        os.makedirs(path)
    for i in range(images.shape[0]):
        img_name = filenames[i]
        print("About to save image {} to {}".format(img_name, path))
        filename = os.path.join(path, img_name)
        image = normalize(images[i]) if normalize_img else images[i]
        plt.imsave(filename, image)


def save_images(images, path, normalize_img=True):
    if not os.path.exists(path):
        os.makedirs(path)
    for i in range(images.shape[0]):
        img_name = 'satImage_{}.{}'.format(str(i + 1).zfill(3), 'png')
        filename = os.path.join(path, img_name)
        image = normalize(images[i]) if normalize_img else images[i]
        plt.imsave(filename, image)

        
def flatten(images):
    X = images
    X = X.reshape(X.shape[0], X.shape[1]*X.shape[2]*X.shape[3])
    return X


def unflatten(X, original_shape):
    n = X.shape[0]
    images = np.zeros((n, *original_shape))
    for i in range(n):
        images[i] = X[i].reshape(*original_shape)
    return images


def zca(X, epsilon=0.1):
    X_norm = X
    X_norm = X_norm - X_norm.mean(axis=0)
    cov = np.cov(X_norm, rowvar=True)
    U, S, _ = np.linalg.svd(cov)
    X_ZCA = U.dot(np.diag(1.0/np.sqrt(S + epsilon))).dot(U.T).dot(X_norm)
    return X_ZCA

def normalize(image):
    """Normalizes the image to the range [0,1]
    
    Args:
        image (ndarray): The image to perform the normalization on
    
    Returns:
        (ndarray): The normalized image
    """
    img_min = image.min()
    img_max = image.max()
    return (image - img_min) / (img_max - img_min)


if __name__ == '__main__':
    # Create whitened training images/labels:
    create_whitened_images(training_images_path, w_training_images_path)
    create_whitened_labels(training_labels_path, w_training_labels_path)
    # Create whitened validation images/labels:
    create_whitened_images(validation_images_path, w_validation_images_path)
    create_whitened_labels(validation_labels_path, w_validation_labels_path)
    # Create whitened test images:
    create_whitened_test_images(test_images, w_test_images)
