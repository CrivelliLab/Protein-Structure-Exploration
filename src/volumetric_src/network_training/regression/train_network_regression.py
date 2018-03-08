'''
train_network_regression.py
Updated: 2/28/17

This script is used to train keras nueral networks using a defined HDF5 file.
Model weights with highest training validation accuracy will be saved.

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

    # Shuffle train data
    train_set = f['train']
    classes = list(train_set.keys())

    x_train = []
    y_train = []
    y_scores = []
    for i in range(len(classes)):
        x = [name for name in train_set[classes[i]] if name.endswith(dim_type)]
        y_score = [scores_dict[j[:-3]] for j in x]
        y = [i for j in range(len(x))]
        x_train += x
        y_train += y
        y_scores += y_score
    x_train = np.expand_dims(x_train, axis=-1)
    y_train = np.expand_dims(y_train, axis=-1)
    y_scores = np.expand_dims(y_scores, axis=-1)
    train = np.concatenate([x_train,y_train,y_scores],axis=-1)
    np.random.seed(seed)
    np.random.shuffle(train)

    # Shuffle validation data
    val_set = f['val']
    classes = list(val_set.keys())

    x_val = []
    y_val = []
    y_val_scores = []
    for i in range(len(classes)):
        x = [name for name in val_set[classes[i]] if name.endswith(dim_type)]
        y_score = [scores_dict[j[:-3]] for j in x]
        y = [scores_dict[j[:-3]] for j in x]
        x_val += x
        y_val += y
        y_val_scores += y_score
    x_val = np.expand_dims(x_val, axis=-1)
    y_val = np.expand_dims(y_val, axis=-1)
    y_val_scores = np.expand_dims(y_val_scores, axis=-1)
    val = np.concatenate([x_val,y_val,y_val_scores],axis=-1)
    np.random.seed(seed)
    np.random.shuffle(val)


    # Load Model
    model, loss, optimizer, metrics = model_def(nb_chans)
    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
    model.summary()

    # Training Loop
    history = []
    best_val_loss = 0.0
    for epoch in range(epochs):
        print("Epoch", epoch, ':')

        # Fit training data
        print('Fitting:')
        train_status = []
        for i in tqdm(range(len(train))):
            x = np.array(train_set[classes[int(train[i,1])]+'/'+train[i,0]])
            x = np.expand_dims(x, axis=0)
            y = np.expand_dims(train[i,2], axis=0)
            output = model.train_on_batch(x, y)
            train_status.append(output)

        # Calculate training loss and accuracy
        train_status = np.array(train_status)
        train_loss = np.average(train_status[:,0])
        train_acc = np.average(train_status[:,1])
        print('Train Loss ->',train_loss)
        print('Train Accuracy ->',train_acc,'\n')

        # Test on validation data
        print('Evaluating:')
        val_status = []
        for i in tqdm(range(len(val))):
            x = np.array(val_set[classes[int(val[i,1])]+'/'+val[i,0]])
            x = np.expand_dims(x, axis=0)
            y = np.expand_dims(val[i,2], axis=0)
            output = model.test_on_batch(x, y)
            val_status.append(output)

        # Calculate validation loss and accuracy
        val_status = np.array(val_status)
        val_loss = np.average(val_status[:,0])
        val_acc = np.average(val_status[:,1])
        print('Val Loss ->',val_loss)
        print('Val Accuracy ->',val_acc,'\n')

        if val_loss > best_val_loss:
            best_val_loss = val_loss

            # Save weights of model
            model.save_weights(data_folder+model_def.__name__+'.hdf5')

        history.append([epoch, train_loss, train_acc, val_loss, val_acc])

    # Parse test data
    test_set = f['test']
    classes = list(test_set.keys())

    x_test = []
    y_test = []
    y_test_scores = []
    for i in range(len(classes)):
        x = [name for name in test_set[classes[i]] if name.endswith(dim_type)]
        y = [scores_dict[j[:-3]] for j in x]
        y_score = [scores_dict[j[:-3]] for j in x]
        x_test += x
        y_test += y
        y_test += y_score
    x_test = np.expand_dims(x_test, axis=-1)
    y_test = np.expand_dims(y_test, axis=-1)
    y_test_scores = np.expand_dims(y_test_scores, axis=-1)
    test = np.concatenate([x_test,y_test,y_test_scores],axis=-1)

    # Load weights of best model
    model.load_weights(model_folder+model_def.__name__+'.hdf5')

    # Evaluate test data
    print('Evaluating Test:')
    test_status = []
    for i in tqdm(range(len(test))):
        x = np.array(test_set[classes[int(test[i,1])]+'/'+test[i,0]])
        x = np.expand_dims(x, axis=0)
        y = np.expand_dims(test[i,2], axis=0)
        y = np.expand_dims(y, axis=0)
        output = model.test_on_batch(x, y)
        test_status.append(output)

    # Calculate test loss and accuracy
    test_status = np.array(test_status)
    test_loss = np.average(test_status[:,0])
    test_acc = np.average(test_status[:,1])
    print('Test Loss ->',test_loss)
    print('Test Accuracy ->',test_acc,'\n')

    # Save training history to csv file
    history = np.array(history)
    test_footer = 'Test [loss, acc]: ' + str(test_loss) + ', ' + str(test_acc)
    np.savetxt(model_folder+model_def.__name__+'.csv', history, fmt= '%1.3f',
                delimiter=', ', header='LABELS: epoch, loss, acc, val_loss, val_acc',
                footer=test_footer)
