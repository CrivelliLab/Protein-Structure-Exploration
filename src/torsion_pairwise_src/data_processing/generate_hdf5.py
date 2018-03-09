'''
generate_hdf5.py
Updated: 3/7/18

'''
import os
import h5py as hp
import numpy as np

# Data folder path
data_folder = '../../../data/KrasHras/'

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
                if os.path.exists(data_folder+class_fn+'/'+data_fn+'/'+data_fn+'.npz'):
                    x_data.append(data_folder+class_fn+'/'+data_fn+'/'+data_fn)
                    y_data.append(class_fn)

    x_data = np.array(x_data)
    y_data = np.array(y_data)

    # Setup HDF5 file and dataset
    print("Storing to HDF5 file...")
    f = hp.File(data_folder+"torsion_pairwise_data.hdf5", "w")

    # Write training data
    grp = f.create_group("dataset")
    for c in classes:
        c_grp = grp.create_group(c)
        for i in range(len(x_data)):
            if y_data[i] == c:
                torsion = np.load(x_data[i]+'.npz')['arr_0']
                dset = c_grp.create_dataset(x_data[i].split('/')[-1]+'-torsion', torsion.shape, dtype='f')
                dset[:,:,:] = torsion[:,:,:]
                pairwise = np.load(x_data[i]+'.npz')['arr_1']
                dset = c_grp.create_dataset(x_data[i].split('/')[-1]+'-pairwise', pairwise.shape, dtype='f')
                dset[:,:,:] = pairwise[:,:,:]
