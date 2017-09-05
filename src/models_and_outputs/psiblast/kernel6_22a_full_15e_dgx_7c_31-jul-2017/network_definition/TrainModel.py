'''
TrainModel.py
Updated: 07/31/17

[PASSING]

README:

The following script is used to train networks on 2D PDB encodings.

Global variables used during training are defined under #- Global Variables.
'data_folder_name' defines the folder containing segemented encoded PDBs. Folders must
be under data/processed/datasets/ .

'network' defines the keras neural network which will be trained.
Note: Network must be imported.

'image-size' defines the shape of the images used for training.
'seed' defines the value used for random number generation.

This script saves a history dictionary to a file called 'train_history_dict' as
well as loss information to a csv named 'loss_history.csv'.

In order to make use of this script for training on new data you must specify
the correct data folder name, the correct number of gpus to be used in
training, the correct image dimensions, the correct batch sizes and
corresponding training and validation steps, and the number of epochs you wish
to train for. 

TODO:
    Turn this into a master script that allows for more automated setting of
    run parameters. Right now this is very manual. 

'''
import os, sys
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
import cPickle as pickle

# Networks
from CIFAR_512 import CIFAR_512

#- Global Variables
network = CIFAR_512(nb_channels=3, nb_class=2, nb_gpu=7)
data_folder_name = 'psiblast/HH-512-MS-FULL'
image_size = (512, 512)
seed = 125

# Verbose Settings
debug = True

################################################################################


if __name__ == '__main__':

    # File Paths
    os.chdir(os.path.dirname(os.path.realpath(__file__)))
    data_folder = "../../../../../data/processed/datasets/" + data_folder_name
    output_folder = '../outputs/'

    # Intiate Keras Flow From Directory
    datagen = ImageDataGenerator()
    train_flow = datagen.flow_from_directory(data_folder +'/train',
                target_size=image_size, batch_size=21, class_mode='categorical',
                seed=seed)
    test_flow = datagen.flow_from_directory(data_folder +'/test',
                target_size=image_size, batch_size=21, class_mode='categorical',
                seed=seed)
    weights_save = ModelCheckpoint(output_folder + 'weights.hdf5', verbose=1, save_weights_only=True, period=5)


    # Fit Training Data
    if debug: print "Training Network..."
    history = network.model.fit_generator(train_flow, epochs=25,
            steps_per_epoch=6785,
                validation_data=test_flow, callbacks=[weights_save,],
                validation_steps=1695)

    # Save Training History
    loss_history = history.history["loss"]
    loss_history = np.array(loss_history)
    np.savetxt(output_folder+"loss_history.csv", loss_history, delimiter=',')

    with open(output_folder + 'train_history_dict', 'wb') as pf:
        pickle.dump(history.history, pf)
