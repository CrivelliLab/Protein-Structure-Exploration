'''
TrainModel.py
Updated: 31 July 2017
[PASSING]

README:

    The following script is used to train networks on 2D PDB encodings, save
    the resulting trained network as a weights file, and create a pickled
    training history object to allow for visualizing network performance. 

    In order to use this script you must change several fields to match your
    data. Important fileds are the 'data_folder_name', 'nb_gpu', 'batch_size',
    'steps_per_epoch', and 'validation_steps.'
'''
# *****************************************************************************
# Imports
# *****************************************************************************
import os, sys
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
import pickle

# Network import
from CIFAR_512 import CIFAR_512

# *****************************************************************************
# Global Variables
# *****************************************************************************
network = CIFAR_512(nb_channels=1, nb_class=13, nb_gpu=2)
data_folder_name = 'shapenet/EncodedShapeNetVox32' # Must match dataset directory structure.
image_size = (512, 512) # Resolution of input images. 
seed = 125 # Random seed for reproducibility. 

# Verbose Settings
debug = True
# *****************************************************************************
# Training Execution
# *****************************************************************************

if __name__ == '__main__':

    # File Paths
    os.chdir(os.path.dirname(os.path.realpath(__file__)))
    data_folder = "../../../../../data/processed/datasets/" + data_folder_name
    output_folder = '../outputs/'

    # Intiate Keras Flow From Directory
    datagen = ImageDataGenerator()
    train_flow = datagen.flow_from_directory(data_folder +'/train',
                target_size=image_size, batch_size=25, class_mode='categorical',
                seed=seed)
    test_flow = datagen.flow_from_directory(data_folder +'/test',
                target_size=image_size, batch_size=25, class_mode='categorical',
                seed=seed)
    weights_save = ModelCheckpoint(output_folder + 'weights.hdf5', verbose=1,
            save_weights_only=True, period=5)

    # Fit Training Data
    if debug: print "Training Network..."
    history = network.model.fit_generator(train_flow, epochs=100,
            steps_per_epoch=1401,
                validation_data=test_flow, callbacks=[weights_save,],
                validation_steps=350)

    # Save Training History
    loss_history = history.history["loss"]
    loss_history = np.array(loss_history)
    np.savetxt(output_folder + "loss_history.csv", loss_history, delimiter=',')

    with open(output_folder + 'train_history_dict.p', 'wb') as pf:
        pickle.dump(history.history, pf)
