# %matplotlib inline
from sklearn import linear_model
import matplotlib.image as mpimg
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from PIL import Image
from sklearn.metrics import confusion_matrix

# Helper functions
def load_image(infilename):
    data = mpimg.imread(infilename)
    return data

def img_float_to_uint8(img):
    rimg = img - np.min(img)
    rimg = (rimg / np.max(rimg) * 255).round().astype(np.uint8)
    return rimg

def img_crop(im, w, h):
    list_patches = []
    imgwidth = im.shape[0]
    imgheight = im.shape[1]
    is_2d = len(im.shape) < 3
    for i in range(0, imgheight, h):
        for j in range(0, imgwidth, w):
            if is_2d:
                im_patch = im[j:j+w, i:i+h]
            else:
                im_patch = im[j:j+w, i:i+h, :]
            list_patches.append(im_patch)
    return list_patches


# Loaded training and validation sets of images
root_dir = r"whitened_images/w_training_preprocessed/training/"  # for whitened training images
val_root_dir = r"whitened_images/w_training_preprocessed/validation/" # for whitened validation images

image_dir = root_dir + "images/"
files = os.listdir(image_dir)
n = len(files)
print("Loading " + str(n) + " images")
imgs = [load_image(image_dir + files[i]) for i in range(n)]

gt_dir = root_dir + "groundtruth/"
print("Loading " + str(n) + " images")
gt_imgs = [load_image(gt_dir + files[i]) for i in range(n)]

val_image_dir = val_root_dir + "images/"
val_files = os.listdir(val_image_dir)
val_n = len(val_files)
print("Loading " + str(val_n) + " images")
val_imgs = [load_image(val_image_dir + val_files[i]) for i in range(val_n)]

val_gt_dir = val_root_dir + "groundtruth/"
print("Loading " + str(val_n) + " images")
val_gt_imgs = [load_image(val_gt_dir + val_files[i]) for i in range(val_n)]


patch_size = 16  # each patch is 16*16 pixels

img_patches = [img_crop(imgs[i], patch_size, patch_size) for i in range(n)]
gt_patches = [img_crop(gt_imgs[i], patch_size, patch_size) for i in range(n)]
val_img_patches = [img_crop(val_imgs[i], patch_size, patch_size) for i in range(val_n)]
val_gt_patches = [img_crop(val_gt_imgs[i], patch_size, patch_size) for i in range(val_n)]

# Linearize list of patches
img_patches = np.asarray([img_patches[i][j] for i in range(
    len(img_patches)) for j in range(len(img_patches[i]))])
gt_patches = np.asarray([gt_patches[i][j] for i in range(
    len(gt_patches)) for j in range(len(gt_patches[i]))])
val_img_patches = np.asarray([val_img_patches[i][j] for i in range(
    len(val_img_patches)) for j in range(len(val_img_patches[i]))])
val_gt_patches = np.asarray([val_gt_patches[i][j] for i in range(
    len(val_gt_patches)) for j in range(len(val_gt_patches[i]))])

# Extract 2-dimensional features consisting of average gray color as well as variance
def extract_features_2d(img):
    feat_m = np.mean(img)
    feat_v = np.var(img)
    feat = np.append(feat_m, feat_v)
    return feat

# Compute features for each image patch
# percentage of pixels > 1 required to assign a foreground label to a patch
foreground_threshold = 0.7  # same as for our u-net

def value_to_class(v):
    df = np.sum(v)
    if df > foreground_threshold:
        return 1
    else:
        return 0

X = np.asarray([extract_features_2d(img_patches[i])
                for i in range(len(img_patches))])
Y = np.asarray([value_to_class(np.mean(gt_patches[i]))
                for i in range(len(gt_patches))])
X_val = np.asarray([extract_features_2d(val_img_patches[i])
                for i in range(len(val_img_patches))])
Y_val = np.asarray([value_to_class(np.mean(val_gt_patches[i]))
                for i in range(len(val_gt_patches))])

# train a logistic regression classifier
# we create an instance of the classifier and fit the data
logreg = linear_model.LogisticRegression(C=1e5, class_weight="balanced", solver="lbfgs")
logreg.fit(X, Y)

# Predict on the validation set
Z_val = logreg.predict(X_val)

cm = confusion_matrix(y_true = Y_val, y_pred = Z_val)
TN = cm[0,0]
FN = cm[1,0]
TP = cm[1,1]
FP = cm[0,1]
print((TN+TP)/(TN+TP+FN+FP)) # accuracy on validation set
