'''
generate_attention.py
Updated: 2/2/17

Script generates saliency map for 2D representation using trained network weights.

'''
import sys; sys.path.insert(0, '../')
import os
import numpy as np
import h5py as hp
from network_training.networks import *
import matplotlib.pyplot as plt
from vis.visualization import visualize_saliency

# Network Training Parameters
model_def = D2NET_v1
nb_chans = 8
nb_layers = 8
weights_path = '../../data/KrasHras/BESTNET.hdf5'

# Data Parameters
data_path = '../../data/KrasHras/Hras/1aa9_A'
classes = 2
class_int = 0

################################################################################

display = True
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
    np.savez(data_path + '/' + data_path.split('/')[-1] + '-2d_sal.npz', atten_map)

    # Display Attention Map
    if display:
        temp_map = np.array([atten_map,atten_map,atten_map])
        temp_map = np.transpose(temp_map, (1,2,0))
        plt.imshow(temp_map)
        plt.show()
