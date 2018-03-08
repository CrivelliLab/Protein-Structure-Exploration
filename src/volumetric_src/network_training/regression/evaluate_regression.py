'''
train_network.py
Updated: 12/27/17

This script is used to evaulate keras nueral networks using a defined HDF5 file.

'''
import os
import numpy as np
import h5py as hp
from tqdm import tqdm
from networks import *
from keras.utils import to_categorical as one_hot

# Network Training Parameters
epochs = 10
model_def = D2NETREG_v1
nb_chans = 8
model_folder = '../../models/T0882_reg/'

# Data Parameters
data_folder = '/tmp/rzamora/T0882_2class/'
scores_file = '../../data/T0882/T0882.csv'
dim_type = '-2d' # -1d, -2d, or -3d

################################################################################

seed = 1234

if __name__ == '__main__':

    # Set paths relative to this file
    os.chdir(os.path.dirname(os.path.realpath(__file__)))

    # Gather ids and GDT-MM scores from scores.txt
    ids = []
    scores = []
    with open(scores_file, 'r') as f:
        lines = f.readlines()
        for i in range(len(lines)):
            x = lines[i].split(',')
            if i > 1 and len(x) == 4:
                id_ = x[0]
                score = float(x[2])
                ids.append(id_)
                scores.append(score)
    scores_dict = dict(zip(ids, scores))

    # Load HDF5 dataset
    f = hp.File(data_folder+"dataset.hdf5", "r")

    # Parse test data
    test_set = f['test']
    classes = list(test_set.keys())

    x_test = []
    y_test = []
    y_test_scores = []
    for i in range(len(classes)):
        x = [name for name in test_set[classes[i]] if name.endswith(dim_type)]
        y = [i for j in range(len(x))]
        y_score = [scores_dict[j[:-3]] for j in x]
        x_test += x
        y_test += y
        y_test_scores += y_score
    x_test = np.expand_dims(x_test, axis=-1)
    y_test = np.expand_dims(y_test, axis=-1)
    y_test_scores = np.expand_dims(y_test_scores, axis=-1)
    test = np.concatenate([x_test,y_test,y_test_scores],axis=-1)

    # Load Model
    model, loss, optimizer, metrics = model_def(nb_chans)
    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
    model.summary()
    model.load_weights(model_folder+model_def.__name__+'.hdf5')

    # Evaluate test data
    print('Evaluating Test:')
    #test_status = []
    for i in tqdm(range(len(test))):
        x = np.array(test_set[classes[int(test[i,1])]+'/'+test[i,0]])
        x = np.expand_dims(x, axis=0)
        y = np.expand_dims(test[i,2], axis=0)
        y = np.expand_dims(y, axis=0)
        output = model.predict_on_batch(x)
        print(test[i,0], y[0], output[0])
        #test_status.append(output)

    # Calculate test loss and accuracy
    #test_status = np.array(test_status)
    #test_loss = np.average(test_status[:,0])
    #test_acc = np.average(test_status[:,1])
    #print('Test Loss ->',test_loss)
    #print('Test Accuracy ->',test_acc,'\n')
