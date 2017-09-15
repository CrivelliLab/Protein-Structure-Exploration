'''
split_data.py
Updated: 9/12/17

README:
The following script is used to split datasets into training and testing.
Global variables used during training are defined under #- Global Variables.

'''
import os, sys
from shutil import copyfile
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split

#- Global Variables
data_folder = '../../data/raw/'
split_folder = '../../data/split/'
file_type  = '.png'

num_augmen = 512
seed = 45
split = [0.7, 0.2, 0.1]

################################################################################

if __name__ == '__main__':

    # File Paths
    os.chdir(os.path.dirname(os.path.realpath(__file__)))
    if not os.path.exists(split_folder): os.mkdir(split_folder)
    if not os.path.exists(split_folder + "/train"): os.mkdir(split_folder + "/train")
    if not os.path.exists(split_folder + "/validation"): os.mkdir(split_folder + "/validation")
    if not os.path.exists(split_folder + "/test"): os.mkdir(split_folder + "/test")

    # Load Training Data
    for folder in sorted(os.listdir(data_folder)):
        if not os.path.isdir(data_folder+folder): continue

        print("Processing", folder, "...")

        # Read File Names
        data_files = []
        for line in sorted(os.listdir(data_folder+folder)):
            if line.endswith(file_type):
                if num_augmen > 1:
                    e = line.split('-')[0]
                    if e not in data_files: data_files.append(e)
                else: data_files.append(line)
        data_files = np.array(data_files)

        # Take Sub Sample of Files
        if sample:
            np.random.seed(seed)
            np.random.shuffle(data_files)
            data_files = data_files[:sample]

        print("Splitting Training and Test Data...")

        # Random Train/Test Split
        x_data, x_test, y_data, y_test = train_test_split(data_files, data_files, test_size=split[1]+split[2], random_state=seed)
        x_val, x_test, y_val, y_test = train_test_split(data_files, data_files, test_size=split[2]/(split[1]+split[2]), random_state=seed)

        print("Copying Training Data...")
        os.mkdir(split_folder + "/train/" + folder)
        for j in tqdm(range(len(x_data))):
            if num_augmen > 1:
                for i in range(num_augmen):
                    fn = x_data[j] + '-r' + str(i) + file_type
                    copyfile(data_folder+folder + '/' + fn, split_folder + 'train/' + folder + '/' + fn)
            else:
                copyfile(data_folder+folder + '/' + x_data[j], split_folder + 'train/' + folder + '/' + x_data[j])

        print("Copying Validation Data...")
        os.mkdir(split_folder + "/validation/" + folder)
        for j in tqdm(range(len(x_val))):
            if num_augmen > 1:
                for i in range(num_augmen):
                    fn = x_val[j] + '-r' + str(i) + file_type
                    copyfile(data_folder+folder + '/' + fn, split_folder + 'validation/' + folder + '/' + fn)
            else:
                copyfile(data_folder+folder + '/' + x_val[j], split_folder + 'validation/' + folder + '/' + x_val[j])

        print("Copying Testing Data...")
        os.mkdir(split_folder + "/test/" + folder)
        for j in tqdm(range(len(x_test))):
            if num_augmen > 1:
                for i in range(num_augmen):
                    fn = x_test[j] + '-r' + str(i) + file_type
                    copyfile(data_folder+folder + '/' + fn, split_folder + 'test/' + folder + '/' + fn)
            else:
                copyfile(data_folder+folder + '/' + x_test[j], split_folder + 'test/' + folder + '/' + x_test[j])
