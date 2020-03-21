"""
Performs baggin on a hand selected collection of prediction
directories and outputs them in a specified path.
Put all folders with predictions of the test images inside
'results/selected_folders_for_bagging/' and run this script
as 'python bagging.py'
"""

import os
import numpy as np
import matplotlib.pyplot as plt

def bagging(input_prediction_paths, output_predictions_path):
    num_models = len(input_prediction_paths)
    num_predictions = 94
    height = 608
    width = height

    input_predictions = np.zeros((num_models, num_predictions, height, width))

    # load the predictions:
    for model_num in range(num_models):
        print(
            "{}/{}: loading predictions from model = {} ..."
            .format(
                str(model_num + 1),
                str(num_models), 
                input_prediction_paths[model_num]
            )
        )
        images_model = os.listdir(input_prediction_paths[model_num])
        for image_num in range(num_predictions):
            image_name = images_model[image_num]
            input_predictions[model_num, image_num, :, :] = plt.imread(
                input_prediction_paths[model_num] + image_name
            )[:,:,0]
        print('done')
    input_predictions[input_predictions >= 0.5] = 1
    input_predictions[input_predictions < 0.5] = 0

    pixel_sums_over_models = np.mean(input_predictions, axis=0)

    output_predictions = np.zeros((num_predictions, height, width))
    output_predictions[pixel_sums_over_models >= 0.5] = 1
    output_predictions[pixel_sums_over_models < 0.5] = 0

    print('Saving predictions in ' + output_predictions_path + ' ...')
    images = os.listdir(input_prediction_paths[0])
    for prediction in range(num_predictions):
        filename = images[prediction]
        plt.imsave(
            output_predictions_path + filename, 
            output_predictions[prediction, :, :], 
            cmap='gray'
        )
    print('done')



if __name__ == '__main__':
    input_prediction_paths = []
    # add an append for every results folder to be included
    prediction_folders = os.listdir('results/selected_folders_for_bagging')
    for i in range(len(prediction_folders)):
        dir_name = prediction_folders[i]
        print(dir_name)
        input_prediction_paths.append(
            "results/selected_folders_for_bagging/{}/"
            .format(dir_name)
        )
    #the directory where we store the bagged predictions:
    output_predictions_path = 'results/bagged_predictions/'

    # Create directory if it doesn't exist yet.
    if not os.path.exists(output_predictions_path):
        print("Creating directory {}".format(output_predictions_path))
        os.makedirs(output_predictions_path)

    bagging(input_prediction_paths, output_predictions_path)
    print('All done')
