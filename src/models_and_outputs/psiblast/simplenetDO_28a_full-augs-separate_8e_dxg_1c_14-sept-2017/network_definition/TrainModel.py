'''
TrainModel.py
Updated: 12 September 2017
[PASSING]

README:

    The following script is used to train networks on 2D PDB encodings, save
    the resulting trained network as a weights file, and create a pickled
    training history object to allow for visualizing network performance. 
    The script will check for prevously-trained weights located at 
    '../outputs/weights.hdf5' and will load them if they are available. 

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
from keras.callbacks import CSVLogger
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
import pickle

# Network import
from SIMPLENET_DROPOUT import SIMPLENET_DROPOUT

# *****************************************************************************
# Global Variables
# *****************************************************************************
network = SIMPLENET_DROPOUT(nb_channels=3, nb_class=2, nb_gpu=1)
data_folder_name = 'psiblast/HH-512-MS-FULL-SEPARATE-AUGMENTS' # Must match dataset directory structure.
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
    data_path = "../../../../../data/processed/datasets/" + data_folder_name
    output_folder = '../outputs/'

    # Intiate Keras Flow From Directory
    datagen = ImageDataGenerator()
    train_flow = datagen.flow_from_directory(data_path +'/train',
                target_size=image_size, batch_size=4, class_mode='categorical',
                seed=seed)
    test_flow = datagen.flow_from_directory(data_path +'/test',
                target_size=image_size, batch_size=4, class_mode='categorical',
                seed=seed)
    # Every three epochs save the weights if they are better than previous. 
    weights_save = ModelCheckpoint(output_folder + 'weights.hdf5',
            monitor='val_acc',  verbose=1, save_best_only=True,
            save_weights_only=True, mode=max, period=3)
    # Streams epoch results to a .csv file - appends to preexisting file. 
    csv_logger = CSVLogger(output_folder + 'training_log.csv', separator=',', append=True)
    # Stop the training if validation accuracy stops improving.
    early_stopper = EarlyStopping(monitor='val_loss', min_delta=.001, patience=7,
            mode='auto')

    # Check for the existance of a previous weights file, load if present.
    if os.path.isfile(output_folder + 'weights.hdf5'):
        print "Resuming from weights file found at " + output_folder + "weights.hdf5'."
        network.model.load_weights(output_folder + 'weights.hdf5')

    # Fit Training Data
    if debug: print "Training Network..."
    history = network.model.fit_generator(train_flow, epochs=100,
            steps_per_epoch=33408,
                validation_data=test_flow, callbacks=[weights_save, csv_logger,
                    early_stopper,], validation_steps=11136)

    # Save Training History - currently these only save after training
    # completes.
    loss_history = history.history["loss"]
    loss_history = np.array(loss_history)
    np.savetxt(output_folder + "loss_history.csv", loss_history, delimiter=',')

    with open(output_folder + 'train_history_dict.p', 'wb') as pf:
        pickle.dump(history.history, pf)
