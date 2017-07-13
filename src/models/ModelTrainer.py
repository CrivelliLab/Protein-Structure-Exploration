'''
ModelTrainer.py
Updated: 07/12/17
[NOT PASSING] - Unimplemented Functionality
                |- Saving Training History to File

README:

The following script is used to train networks on 2D PDB encodings.

Global variables used during training are defined under #- Global Variables.
'data_folders' defines the folder containing segemented encoded PDBs. Folders must
be under data/processed/datasets/ .

'network' defines the keras neural network which will be trained.
Note: Network must be imported.

'image-size' defines the shape of the images used for training.
'seed' defines the value used for random number generation.

'''
import os, sys
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint

# Networks
from CIFAR_512 import CIFAR_512

#- Global Variables
network = CIFAR_512(nb_channels=3, nb_class=2)
data_folder = ''
image_size = (512, 512)
seed = 125

# Verbose Settings
debug = True

################################################################################

if __name__ == '__main__':

    # File Paths
    data_folder = "../../data/processed/"+ data_folder
    model_folder = '../../models/' + network.__name__ + '/'

    # Intiate Keras Flow From Directory
    datagen = ImageDataGenerator()
    train_flow = datagen.flow_from_directory(data_folder +'/train',
                target_size=image_size, batch_size=2, class_mode='categorical',
                seed=seed)
    test_flow = datagen.flow_from_directory(data_folder +'/test',
                target_size=image_size, batch_size=2, class_mode='categorical',
                seed=seed)
    save = ModelCheckpoint( model_folder + 'weights.hdf5', verbose=1, save_weights_only=True, period=5)

    # Fit Training Data
    if debug: print "Training Network..."
    history = network.model.fit_generator(train_flow, epochs=100, steps_per_epoch=7000,
                validation_data=test_flow, callbacks=[save,], validation_steps=3000)
