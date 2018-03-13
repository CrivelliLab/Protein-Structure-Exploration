'''
train_binary_network.py
Updated: 12/27/17

'''
import os
import h5py as hp
import numpy as np
from tqdm import tqdm
from networks import *
from keras.utils import to_categorical as one_hot
from sklearn.model_selection import train_test_split

# Network Training Parameters
epochs = 10
model_def = PairwiseNet_v1
model_folder = '../../../../models/T0867/'
threshold = 0.8

# Data Parameters
data_folder = '../../../../data/T0867/'
data_type = '-pairwise' # '-pairwise', '-torsion'
split = [0.7, 0.1, 0.2]
seed = 678452

################################################################################

if __name__ == '__main__':

    # Set paths relative to this file
    os.chdir(os.path.dirname(os.path.realpath(__file__)))

    # Gather ids and GDT-MM scores from csv
    ids = []
    scores = []
    with open(data_folder+data_folder.split('/')[-2]+'.csv', 'r') as f:
        lines = f.readlines()
        for i in range(len(lines)):
            x = lines[i].split(',')
            if i > 1 and len(x) == 4:
                id_ = x[0]; score = float(x[3])
                ids.append(id_); scores.append(score)
    x_data = np.array(ids)
    scores = np.array(scores)

    # Split into binary classification problem.
    print('Forming Binary Classification At Threshold:', threshold)
    y_data = []
    for i in scores:
        if i >= threshold: y_data.append(1)
        else: y_data.append(0)
    y_data = np.array(y_data)
    print('Positive:',len(np.where(y_data == 1)[0]),'Negative:', len(np.where(y_data == 0)[0]))

    # Split file paths into training, test and validation
    print("Splitting Training and Test Data...")
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=split[1]+split[2], random_state=seed)
    x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, test_size=split[2]/(split[1]+split[2]), random_state=seed)

    # Load Model
    model, loss, optimizer, metrics = model_def(2)
    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
    model.summary()
    model_path = model_folder+model_def.__name__+'_'+str(threshold)+data_type

    # Load HDF5 dataset
    f = hp.File(data_folder+"torsion_pairwise_casp_data.hdf5", "r")
    data_set = f['dataset']

    # Training Loop
    history = []
    best_val_loss = None
    for epoch in range(epochs):
        print("Epoch", epoch, ':')

        # Fit training data
        print('Fitting:')
        train_status = []
        for i in tqdm(range(len(x_train))):
            x = np.array(data_set[x_train[i]+data_type])
            x = np.expand_dims(x, axis=0)
            y = one_hot(y_train[i], num_classes=2)
            y = np.expand_dims(y, axis=0)
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
        for i in tqdm(range(len(x_val))):
            x = np.array(data_set[x_val[i]+data_type])
            x = np.expand_dims(x, axis=0)
            y = one_hot(y_val[i], num_classes=2)
            y = np.expand_dims(y, axis=0)
            output = model.train_on_batch(x, y)
            val_status.append(output)

        # Calculate validation loss and accuracy
        val_status = np.array(val_status)
        val_loss = np.average(val_status[:,0])
        val_acc = np.average(val_status[:,1])
        print('Val Loss ->',val_loss)
        print('Val Accuracy ->',val_acc,'\n')

        if best_val_loss == None or val_loss < best_val_loss:
            best_val_loss = val_loss

            # Save weights of model
            model.save_weights(model_path+'.hdf5')

        history.append([epoch, train_loss, train_acc, val_loss, val_acc])

    # Load weights of best model
    model.load_weights(model_path+'.hdf5')

    # Evaluate test data
    print('Evaluating Test:')
    test_status = []
    for i in tqdm(range(len(x_test))):
        x = np.array(data_set[x_test[i]+data_type])
        x = np.expand_dims(x, axis=0)
        y = one_hot(y_test[i], num_classes=2)
        y = np.expand_dims(y, axis=0)
        output = model.train_on_batch(x, y)
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
    np.savetxt(model_path+'.csv', history, fmt= '%1.3f', delimiter=', ',
               header='LABELS: epoch, loss, acc, val_loss, val_acc',
               footer=test_footer)
