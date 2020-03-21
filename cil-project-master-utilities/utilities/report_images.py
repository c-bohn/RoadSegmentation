"""Collection of functions for generating informative images
that were used in the report.

Most notably, 'create_overlay_image' overlays a prediction over
the original image to visualize how good the prediction was.

The function 'create_difference_image' performs the absolute difference
between a groundtruth image and the corresponding prediction.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from data_paths import (
    w_validation_images_path,
    w_validation_labels_path
)
from mask_to_submission import patch_to_label
from config import validation_images_path


def create_overlay_image(
        image_path, 
        prediction_name, 
        save_dir, 
        overlay_name,
        mask_color,
        mask_intensity,
        grid=False,
        normalize_img=True):
    """ Takes an image and a prediction and saves a new image to disk, with the 
    prediction overlayed on top of the image
    
    Args:
        image_path (string): Full path of the image (directory + filename)
        prediction_name (string): Full path of the prediction (directory + filename)
        save_dir (string): Directory to where you want to store the produced overlay image.
        (path is realtive to project root)
        overlay_name (string): The name of the produced image. You have to include the file 
            extension ".png" in the name!
        mask_color: 3-tupel, desribing the color of the overlayed prediction in RGB values.
            Every entry must be in [0, 1]. (Example: (0.8, 0.0, 0.8) would be something purple-ish)
        mask_intensity ([type]): Value in [0, 1] describing the alpha value of the prediction mask.
            0 being totally transparent (invisible), 1 being totally opaque (i.e. totally blocking of the
            roads where theprediction is 1).
        normalize_img (bool, optional): [description]. Makes sure that the values stay in [0, 1]
            Defaults to True.
    """
    image = plt.imread(image_path)[:,:,:3]
    prediction = plt.imread(prediction_name)[:,:,:3]
    
    if grid:
        prediction = gridify(prediction)

    overlay = overlay_prediction(image, prediction, mask_color, mask_intensity)
    save_image(overlay, overlay_name, save_dir, normalize_img=normalize_img)

def create_overlay_images(
        images_dir, 
        predictions_dir, 
        save_dir,
        mask_color,
        mask_intensity,
        overlay_prefix='overlay',
        grid=False,
        normalize_img=True):
    """Creates the overlays for an entire directory. Uses 'create_overlay_image' from
    above internally.
    
    Args:
        images_dir ([type]): Path of the directory containing road images.
        predictions_dir ([type]): Path of the directory containing the predictions
        save_dir ([type]): [description]
        mask_color ([type]): 3-tupel, desribing the color of the overlayed prediction in RGB values.
        Every entry must be in [0, 1]. (Example: (0.8, 0.0, 0.8) would be something purple-ish)
        mask_intensity ([type]): Value in [0, 1] describing the alpha value of the prediction mask.
        0 being totally transparent (invisible), 1 being totally opaque (i.e. totally blocking of the
        roads where theprediction is 1).
        normalize_img (bool, optional): [description]. Makes sure that the values stay in [0, 1]
        Defaults to True.
    """
    img_names = os.listdir(images_dir)
    prediction_names = os.listdir(predictions_dir)

    img_names.sort()
    prediction_names.sort()
    for image_name, prediction_name in zip(img_names, prediction_names):
        image_path = os.path.join(images_dir, image_name)
        prediction_name = os.path.join(predictions_dir, prediction_name)
        overlay_name = "{}_{}".format(overlay_prefix, image_name)
        create_overlay_image(
            image_path, 
            prediction_name, 
            save_dir,
            mask_color=mask_color,
            mask_intensity=mask_intensity,
            overlay_name=overlay_name, 
            grid=grid,
            normalize_img=normalize_img
        )


def create_difference_image(mask, prediction):
    """Creates the pixel-wise absolute difference of the groundtruth
    image 'mask' and the prediction image 'prediction'.
    
    Args:
        mask (ndarray): The groundtrush image.
        prediction (ndarray): The prediction image.
    
    Returns:
        [ndarray]: The pixel-wise absolute difference of the images.
    """
    abs_diff = np.abs(mask - prediction)
    return abs_diff


def create_difference_images(masks_dir, predictions_dir, save_dir):
    """Creates the pixel wise absolute difference of an entire directory
    of images. See 'create_difference_image' for more details
    """
    mask_filenames = os.listdir(masks_dir)
    prediction_filenames = os.listdir(predictions_dir)
    assert len(mask_filenames) == len(prediction_filenames)
    mask_filenames.sort()
    prediction_filenames.sort()
    for i in range(len(mask_filenames)):
        mask = plt.imread(masks_dir + mask_filenames[i])[:,:,0]
        prediction = plt.imread(predictions_dir + prediction_filenames[i])[:,:,0]
        diff_img = create_difference_image(mask, prediction)
        save_image(diff_img, 'diff_' + mask_filenames[i], save_dir, cmap='gray')

    

def save_image(image, name, path, normalize_img=True, cmap=None):
    """Saves an image held in a numpy array to disc.
    
    Args:
        image (ndarray): The image to save.
        name (string): The filename of the saved image.
        path (string): The path where the image will be saved.
        normalize_img (bool, optional): If true, values will be normalized 
        to [0, 1]. Defaults to True.
        cmap (string, optional): The colormap to use when saving the image.
        Defaults to None.
    """
    print(name)
    if not os.path.exists(path):
        os.makedirs(path)
    filename = os.path.join(path, name)
    image = normalize(image) if normalize_img else image
    plt.imsave(filename, image, cmap=cmap)


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




def overlay_prediction(image, prediction, mask_color, mask_intensity=0.5):
    mask = prediction[:, :, 0]
    # Extend 1-channel greyscale to 3-channels (repeat grayscale on each channel)
    mask = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
    # Apply color to mask
    mask *= mask_color
    alpha = mask_intensity
    overlay = np.multiply((1. - mask * alpha), image) + mask * alpha
    return overlay


def validation_error_img(label, prediction, error_fun):
    diff = label - prediction
    return np.vectorize(error_fun)(diff)


def gridify(prediction, patch_size=16):
    grid = np.zeros(prediction.shape)
    print(grid.shape)
    for j in range(0, grid.shape[1], patch_size):
        for i in range(0, grid.shape[0], patch_size):
            patch = prediction[i:i + patch_size, j:j + patch_size]
            label = patch_to_label(patch)
            grid[i:i + patch_size, j:j + patch_size] = label
    return grid



def file_iterator(*dirs):
    print(*dirs)
    file_names_list = map(os.listdir, dirs)
    sorted_file_names_list = [file_names.sort() for file_names in file_names_list]
    print(*sorted_file_names_list)
    return zip(sorted_file_names_list)



if __name__ == '__main__':
    """IMPORTANT: 
    You have to define the variables 'images_dir', 'predictions_dir' and 'save_dir' 
    yourself!
    """

    #/TODO 1. Specify where to load the images from
    images_dir = validation_images_path

    #/TODO 2. Specfiy where to load the predict4ions from
    predictions_dir = 'results/predictions_on_val_set/'

    #/TODO 3. Specify where to store the overlays
    save_dir = 'results/overlays/'#os.path.join('report')

    #/TODO 4. Define color and alpha values for mask
    mask_color = (1, 0, 1)
    mask_intensity = 0.4

    # #/TODO 5. That's it. Run the script and check the path you specified in
    # #  'save_dir'.
    create_overlay_images(
        images_dir,
        predictions_dir,
        save_dir,
        mask_color=mask_color,
        mask_intensity=mask_intensity,
        overlay_prefix='overlay'
    )

    masks_dir = w_validation_labels_path
    difference_imgs_save_dir = 'results/diff_imgs/'
    create_difference_images(
        masks_dir,
        predictions_dir,
        difference_imgs_save_dir
    )

    
    # / TODO (Alternatively) Uncomment this if you want to render only single, 
    # hand-picked image and not render the whole directory. Simply change the id to the image
    # number you like. Steps 1, 2, 3 and 4 are still necesary.

    # id = 93
    #
    # image_name = 'test_{}.png'.format(id)
    # prediction_name = 'prediction_test_{}.png'.format(id)
    #
    # image_path = os.path.join(images_dir, image_name)
    # prediction_name = os.path.join(predictions_dir, prediction_name)
    #
    # create_overlay_image(
    #     image_path,
    #     prediction_name,
    #     save_dir,
    #     'single_overlayed_grid_{}.png'.format(id),
    #     mask_color,
    #     mask_intensity,
    #     grid=True
    # )

