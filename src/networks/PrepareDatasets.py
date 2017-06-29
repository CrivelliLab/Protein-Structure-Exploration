'''
PrepareDatasets.py
Updated: 06/29/17

README:

The following script is used to split datasets into training and testing.

Global variables used during training are defined under #- Global Variables.
data_folders defines the list of folder containing encoded PDBs. Folders must
be under data/final/.

There is a 0.7/0.3 split of the data to generate training and testing data.

'''
import os, sys
from shutil import copyfile
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split

#- Global Variables
data_folders = ['RAS-MD512-HH', 'WD40-MD512-HH']
seed = 45
split = 0.3

# Verbose Settings
debug = True

################################################################################

if __name__ == '__main__':

    if debug: print "Preparing Dataset..."

    folder = ''
    for i in range(len(data_folders)):
        if i != len(data_folders)-1:
            folder += data_folders[i].split('-')[0] + '-'
        else: folder += data_folders[i]
    os.chdir("../../data/final/")
    os.mkdir(folder)
    os.mkdir(folder + "/train")
    os.mkdir(folder + "/test")

    # Load Training Data
    for i in range(len(data_folders)):

        if debug: print "Processing", data_folders[i], "..."

        # Read PDB IMG File Names
        data_files = []
        for line in sorted(os.listdir(data_folders[i] + '/')):
            if line.endswith('.png'): data_files.append(line)
        data_files = np.array(data_files)

        if debug: print "Splitting Training and Test Data..."

        x_data, x_test, y_data, y_test = train_test_split(data_files, data_files, test_size=split, random_state=seed)

        if debug: print "Copying Training Data..."
        os.mkdir(folder + "/train/" + data_folders[i])
        for j in tqdm(range(len(x_data))):
            copyfile(data_folders[i] + '/' + x_data[i], folder + '/train/' + data_folders[i] + '/' + x_data[i])

        if debug: print "Copying Testing Data..."
        os.mkdir(folder + "/test/" + data_folders[i])
        for j in tqdm(range(len(x_test))):
            copyfile(data_folders[i] + '/' + x_test[i], folder + '/test/' + data_folders[i] + '/' + x_test[i])
