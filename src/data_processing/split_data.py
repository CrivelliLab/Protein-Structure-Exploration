'''
split_data.py
Updated: 11/29/17

The following script is used to split datasets into training and testing.

'''
import os, sys
import numpy as np
from tqdm import tqdm
from shutil import copyfile
from sklearn.model_selection import train_test_split

# Data path and types
data_folder = '../../data/raw/KRAS_HRAS/'
split_folder = '../../data/split/KRAS_HRAS_split110117/'
file_type  = '.png'


num_augmen = 500
seed = 102317
split = [0.7, 0.1, 0.2]

################################################################################

if __name__ == '__main__':

    # Set file paths relative to this file
    os.chdir(os.path.dirname(os.path.realpath(__file__)))

    # Create directories if they do not yet exists
    if not os.path.exists(split_folder): os.mkdir(split_folder)
    if not os.path.exists(split_folder + "/train"): os.mkdir(split_folder + "/train")
    if not os.path.exists(split_folder + "/validation"): os.mkdir(split_folder + "/validation")
    if not os.path.exists(split_folder + "/test"): os.mkdir(split_folder + "/test")

    # Search through dataset folder
    for folder in sorted(os.listdir(data_folder)):
        if not os.path.isdir(data_folder+folder): continue
        print("Processing", folder, "...")

        # Search for paths of defined type files within class sub directories of dataset folder
        data_files = []
        for line in sorted(os.listdir(data_folder+folder)):
            if line.endswith(file_type):

                # Add on unique class members
                if num_augmen > 1:
                    e = line.split('-')[0]
                    if e not in data_files: data_files.append(e)
                else: data_files.append(line)
        data_files = np.array(data_files)

        # Split file paths into training, test and validation
        print("Splitting Training and Test Data...")
        x_data, x_test, y_data, y_test = train_test_split(data_files, data_files, test_size=split[1]+split[2], random_state=seed)
        x_val, x_test, y_val, y_test = train_test_split(x_test, x_test, test_size=split[2]/(split[1]+split[2]), random_state=seed)

        # Copy Training files into corresponding folders
        print("Copying Training Data...")
        os.mkdir(split_folder + "/train/" + folder)
        for j in tqdm(range(len(x_data))):
            if num_augmen > 1:
                for i in range(num_augmen):
                    fn = x_data[j] + '-r' + str(i) + file_type
                    try: copyfile(data_folder+folder + '/' + fn, split_folder + 'train/' + folder + '/' + fn)
                    except: pass
            else:
                copyfile(data_folder+folder + '/' + x_data[j], split_folder + 'train/' + folder + '/' + x_data[j])

        # Copy Validation files to corresponding folders
        print("Copying Validation Data...")
        os.mkdir(split_folder + "/validation/" + folder)
        for j in tqdm(range(len(x_val))):
            if num_augmen > 1:
                for i in range(num_augmen):
                    fn = x_val[j] + '-r' + str(i) + file_type
                    try:copyfile(data_folder+folder + '/' + fn, split_folder + 'validation/' + folder + '/' + fn)
                    except: pass
            else:
                copyfile(data_folder+folder + '/' + x_val[j], split_folder + 'validation/' + folder + '/' + x_val[j])

        # Copy Testing files to corresponding folders
        print("Copying Testing Data...")
        os.mkdir(split_folder + "/test/" + folder)
        for j in tqdm(range(len(x_test))):
            if num_augmen > 1:
                for i in range(num_augmen):
                    fn = x_test[j] + '-r' + str(i) + file_type
                    try:copyfile(data_folder+folder + '/' + fn, split_folder + 'test/' + folder + '/' + fn)
                    except: pass
            else:
                copyfile(data_folder+folder + '/' + x_test[j], split_folder + 'test/' + folder + '/' + x_test[j])
