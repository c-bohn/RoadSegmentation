from utilities.data_paths import (
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

""" 
The purpose of this module is to provide all the necessary hyperparameters
to the 'training.py' file, as well as specifying the models to be loaded before
running 'predict.py'. All parameters are stored in a python dictionary named 
'config', defined in the next line:

"""
config = dict()


###############################################################################
### PREDICTION
###############################################################################

# Tweak the following parameters before running 'predict.py'.

config['model_timestamp'] = '20190628_094425'       # The timestamp to specify which model should be loaded for pediction.
config['best_model'] = 1191                         # The number identifying the optimal among the saved models
predictions_path = 'results/my_prediction'          # Where to store the predictions


###############################################################################
### TRAINING
###############################################################################

# Tweak the following hyperparameters before running 'training.py'.

config['model_dir'] = 'model_files/'                # Where to store the trained models.


config['num_epochs'] = 5000                         # The maximum number of epochs to train for
config['patience'] = 300                            # The maximum number of epochs without improvement in validation


config['dropout_rate'] = 0.0                        # The fraction of weights to drop out
config['learning_rate'] = 5*1e-5                    # The learning rate of the optimizer
config['use_exp_learning_rate_decay'] = True        # Whether to use a fixed or exponentially decaying learning rate

config['use_batch_norm'] = False                    # Whether to use batch normalization
config['num_test_imgs'] = 94                        # The number of test images
config['predict_after_training'] = False            # Whether to predict directly after training using the weights after
                                                    # the last epoch

config['class_weight'] = 5.0                        # how to weigh the classes to mitigate the class imbalance
config['spatial_dropout_rate'] = 0.4                # obvious

config['dilation_rate'] = 2

config['whitening'] = True                          # Whether to use the preprocessed whitened images, or the original dataset.
config['postprocessing'] = False                    # If True, the predictions get postprocessed before saved to disc


# Load the data paths depending wheter we want to use the whitended or the original data:
training_images_path = (
    training_images_path if not config['whitening']
    else w_training_images_path
)
training_labels_path = (
    training_labels_path if not config['whitening']
    else w_training_labels_path
)
validation_images_path = (
    validation_images_path if not config['whitening']
    else w_validation_images_path
)
validation_labels_path = (
    validation_labels_path if not config['whitening']
    else w_validation_labels_path
)
test_images_path = (
    test_images if not config['whitening']
    else w_test_images
)

