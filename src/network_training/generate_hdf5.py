'''
generate_hdf5.py
Updated: 12/27/17

Script is used to build HDF5 file which will be used for network training.
HDF5 file will contain 3D, 2D, and 1D represenation of data split into
train, validation and test sets.

'''
import os
import h5py as hp
import numpy as np
from scipy.io import wavfile
from sklearn.model_selection import train_test_split

# Data folder path
data_folder = '../../data/KrasHras/'
nb_rots = 10

# Training split
split = [0.7, 0.1, 0.2]

################################################################################
seed = 1234

if __name__ == '__main__':

    # Set paths relative to this file
    os.chdir(os.path.dirname(os.path.realpath(__file__)))

    # Search for class folders withing dataset folder
    print("Searching class folders for entries...")
    x_data = []
    y_data = []
    classes = []
    for class_fn in sorted(os.listdir(data_folder)):
        if os.path.isdir(data_folder+class_fn):
            classes.append(class_fn)

            # Search for files within class folders and add to list of data
            for data_fn in sorted(os.listdir(data_folder+class_fn)):
                if os.path.exists(data_folder+class_fn+'/'+data_fn+'/'+data_fn+'-3d.npz'):
                    if os.path.exists(data_folder+class_fn+'/'+data_fn+'/'+data_fn+'-2d.npz'):
                        if os.path.exists(data_folder+class_fn+'/'+data_fn+'/'+data_fn+'-1d.npz'):
                            x_data.append(data_folder+class_fn+'/'+data_fn+'/'+data_fn)
                            y_data.append(class_fn)

    x_data = np.array(x_data)
    y_data = np.array(y_data)

    # Split file paths into training, test and validation
    print("Splitting Training and Test Data...")
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=split[1]+split[2], random_state=seed)
    x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, test_size=split[2]/(split[1]+split[2]), random_state=seed)

    # Setup HDF5 file and dataset
    print("Storing to HDF5 file...")
    f = hp.File(data_folder+"dataset.hdf5", "w")

    # Write training data
    grp = f.create_group("train")
    for c in classes:
        c_grp = grp.create_group(c)
        for i in range(len(x_train)):
            if y_train[i] == c:
                array_3d = np.load(x_train[i]+'-3d.npz')['arr_0'].astype('float')
                array_2d = np.load(x_train[i]+'-2d.npz')['arr_0'].astype('float')
                array_1d = np.load(x_train[i]+'-1d.npz')['arr_0'].astype('float')
                dset = c_grp.create_dataset(x_train[i].split('/')[-1]+'-3d', array_3d.shape, dtype='f')
                dset[:,:,:,:] = array_3d[:,:,:,:]
                dset = c_grp.create_dataset(x_train[i].split('/')[-1]+'-2d', array_2d.shape, dtype='f')
                dset[:,:,:] = array_2d[:,:,:]
                dset = c_grp.create_dataset(x_train[i].split('/')[-1]+'-1d', array_1d.shape, dtype='f')
                dset[:,:] = array_1d[:,:]

    # Write test data
    grp = f.create_group("test")
    for c in classes:
        c_grp = grp.create_group(c)
        for i in range(len(x_test)):
            if y_test[i] == c:
                array_3d = np.load(x_test[i]+'-3d.npz')['arr_0'].astype('float')
                array_2d = np.load(x_test[i]+'-2d.npz')['arr_0'].astype('float')
                array_1d = np.load(x_test[i]+'-1d.npz')['arr_0'].astype('float')
                dset = c_grp.create_dataset(x_test[i].split('/')[-1]+'-3d', array_3d.shape, dtype='f')
                dset[:,:,:,:] = array_3d[:,:,:,:]
                dset = c_grp.create_dataset(x_test[i].split('/')[-1]+'-2d', array_2d.shape, dtype='f')
                dset[:,:,:] = array_2d[:,:,:]
                dset = c_grp.create_dataset(x_test[i].split('/')[-1]+'-1d', array_1d.shape, dtype='f')
                dset[:,:] = array_1d[:,:]


    # Write validation data
    grp = f.create_group("val")
    for c in classes:
        c_grp = grp.create_group(c)
        for i in range(len(x_val)):
            if y_val[i] == c:
                array_3d = np.load(x_val[i]+'-3d.npz')['arr_0'].astype('float')
                array_2d = np.load(x_val[i]+'-2d.npz')['arr_0'].astype('float')
                array_1d = np.load(x_val[i]+'-1d.npz')['arr_0'].astype('float')
                dset = c_grp.create_dataset(x_val[i].split('/')[-1]+'-3d', array_3d.shape, dtype='f')
                dset[:,:,:,:] = array_3d[:,:,:,:]
                dset = c_grp.create_dataset(x_val[i].split('/')[-1]+'-2d', array_2d.shape, dtype='f')
                dset[:,:,:] = array_2d[:,:,:]
                dset = c_grp.create_dataset(x_val[i].split('/')[-1]+'-1d', array_1d.shape, dtype='f')
                dset[:,:] = array_1d[:,:]
