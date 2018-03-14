'''
train_ranking_network.py
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
epochs = 3
batch_size = 10
model_def = PairwiseNet_v1
model_folder = '../../../../models/T0882_ranked_1/'

# Data Parameters
data_folder = '../../../../data/T0882/'
ranks = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
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
                id_ = x[0]; score = float(x[2])
                ids.append(id_); scores.append(score)
    x_data = np.array(ids)
    scores = np.array(scores)

    # Split training and test data
    x_data, x_test, y_scores, y_test_scores = train_test_split(x_data, scores, test_size=split[2], random_state=seed)

    # Load HDF5 dataset
    f = hp.File(data_folder+"torsion_pairwise_casp_data.hdf5", "r")
    data_set = f['dataset']

    # Train Rankings
    for i in range(len(ranks)):

        rank = ranks[i]

        # Split into binary classification problem.
        print('Forming Binary Classification At Threshold:', rank)
        y_data = []
        for score in y_scores:
            if score >= rank: y_data.append(1)
            else: y_data.append(0)
        y_data = np.array(y_data)
        print('Positive:',len(np.where(y_data == 1)[0]),'Negative:', len(np.where(y_data == 0)[0]))

        # Split file paths into training, test and validation
        x_train, x_val, y_train, y_val = train_test_split(x_data, y_data, test_size=split[1]/(split[0]+split[1]), random_state=seed)

        if i == 0:
            # Load Model
            model, loss, optimizer, metrics = model_def(2)
            model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
            model.summary()
        else:
            # Load weights of best model
            model.load_weights(model_path+'.hdf5')
        model_path = model_folder+model_def.__name__+'_'+str(rank)+data_type

        # Training Loop
        history = []
        best_val_loss = None
        for epoch in range(epochs):
            print("Epoch", epoch, ':', "Ranking Threshold:", rank)

            # Fit training data
            print('Fitting:')
            train_status = []
            batch_x = []
            batch_y = []
            for j in tqdm(range(len(x_train))):
                x = np.array(data_set[x_train[j]+data_type])
                batch_x.append(x)
                y = one_hot(y_train[j], num_classes=2)
                batch_y.append(y)
                if len(batch_x) == batch_size or j+1 == len(x_train):
                    batch_x = np.array(batch_x)
                    batch_y = np.array(batch_y)
                    output = model.train_on_batch(batch_x, batch_y)
                    batch_x = []
                    batch_y = []
                    train_status.append(output)

            # Calculate training loss and accuracy
            train_status = np.array(train_status)
            train_loss = np.average(train_status[:,0])
            train_acc = np.average(train_status[:,1])
            print('Train Loss ->', train_loss)
            print('Train Accuracy ->', train_acc,'\n')

            # Test on validation data
            print('Evaluating:')
            val_status = []
            batch_x = []
            batch_y = []
            for j in tqdm(range(len(x_val))):
                x = np.array(data_set[x_val[j]+data_type])
                batch_x.append(x)
                y = one_hot(y_val[j], num_classes=2)
                batch_y.append(y)
                if len(batch_x) == batch_size or j+1 == len(x_train):
                    batch_x = np.array(batch_x)
                    batch_y = np.array(batch_y)
                    output = model.test_on_batch(batch_x, batch_y)
                    batch_x = []
                    batch_y = []
                    val_status.append(output)

            # Calculate validation loss and accuracy
            val_status = np.array(val_status)
            val_loss = np.average(val_status[:,0])
            val_acc = np.average(val_status[:,1])
            print('Val Loss ->', val_loss)
            print('Val Accuracy ->', val_acc,'\n')

            if best_val_loss == None or val_loss < best_val_loss:
                best_val_loss = val_loss

                # Save weights of model
                model.save_weights(model_path+'.hdf5')

            history.append([rank, epoch, train_loss, train_acc, val_loss, val_acc])

    # Load Model
    model, loss, optimizer, metrics = model_def(2)
    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
    model.summary()

    # Evaluate test data
    ranked_data = []
    positive_set = x_test
    for i in range(len(ranks)):
        rank = ranks[i]
        positive_set_prime = []
        negative_set_prime = []

        # Load weights of best model
        model_path = model_folder+model_def.__name__+'_'+str(rank)+data_type
        model.load_weights(model_path+'.hdf5')

        # Get inference results
        print("Running Inference On Rank:", rank)
        for j in tqdm(range(len(positive_set))):
            x = np.array(data_set[positive_set[j]+data_type])
            x = np.expand_dims(x, axis=0)
            s = model.predict_on_batch(x)[0]
            s = int(np.argmax(s))
            if s == 1: positive_set_prime.append(positive_set[j])
            else: negative_set_prime.append(positive_set[j])

        positive_set = positive_set_prime
        ranked_data.append(negative_set_prime)
    ranked_data.append(positive_set)
    ranked_data = np.array(ranked_data)

    # Measure accuracy of ranking
    hits = 0
    for i in tqdm(range(len(x_test))):
        x = x_test[i]
        y = y_test_scores[i]

        # Get Rank
        rank = 0
        rankss = ranks + [1.0]
        for j in range(len(rankss)):
            if rankss[j] - y > 0:
                rank = j
                break

        # Check if in correct rank
        if np.isin(x, ranked_data[rank]): hits +=1
    test_acc = float(hits) / len(x_test)
    print("Test Accuracy:", test_acc)

    # Save training history to csv file
    history = np.array(history)
    test_footer = 'Test [acc]: ' + str(test_acc)
    np.savetxt(model_folder+'results.csv', history, fmt= '%1.3f', delimiter=', ',
               header='LABELS: rank, epoch, loss, acc, val_loss, val_acc',
               footer=test_footer)
