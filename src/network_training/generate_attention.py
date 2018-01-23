'''
train_network.py
Updated: 12/27/17

This script is used to train keras nueral networks using a defined HDF5 file.
Best training validation accuracy will be saved.

'''
import os
import numpy as np
import h5py as hp
from networks import *
from keras.utils import to_categorical as one_hot
from vis.visualization import visualize_saliency

# Network Training Parameters
model_def = D2NET
nb_chans = 8
nb_layers = 8
weights_path = '../../data/KrasHras/BESTNET.hdf5'

# Data Parameters
data_path = '../../data/KrasHras/Hras/1clu_A'
classes = 2
class_int = 0
threshold = 0.7

################################################################################

seed = 1234

if __name__ == '__main__':

    # Set paths relative to this file
    os.chdir(os.path.dirname(os.path.realpath(__file__)))

    # Load Model
    model, loss, optimizer, metrics = model_def(nb_chans, classes)
    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
    model.summary()

    # Load weights of best model
    model.load_weights(weights_path)

    # Load data
    array_2d = np.load(data_path + '/' + data_path.split('/')[-1] + '-2d.npz')['arr_0'].astype('int')
    array_2d = np.expand_dims(array_2d, 0)

    # Run inference
    p = model.predict(array_2d, batch_size=1, verbose=1)
    atten_map = visualize_saliency(model, nb_layers, [np.argmax(p[0])], array_2d[0])
    atten_map = atten_map/255.0
    atten_map = np.dot(atten_map[...,:3], [0.299, 0.587, 0.114])
    atten_map[atten_map < threshold] = 0

    import matplotlib.pyplot as plt
    plt.imshow(atten_map)
    plt.show()
