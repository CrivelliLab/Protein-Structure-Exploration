# Generating Datasets
Updated: 1/28/18

The following document details how to generate new protein dataset using repo.

## Step 1: Create Dataset Folder

First, create a new dataset folder in the /data folder. This newly folder will be where
all files for your dataset will be stored. Add .csv files containing pairs of PDB ids and
chain ids for each given class of proteins.

## Step 2: Fetch Needed PDB Files

Open the fetch_pdbs.py file and change the data_folder variable to the path to
the new dataset folder.
```
# Data folder path
data_folder = '../../data/my_dataset/'

###############################################################################

```
Run the fetch_pdbs.py script and wait for all the PDBs to
download. This cript has been parallelized using MPI.

```
$ python src/data_processing/fetch_pdbs.py

or

$ mpirun -n N python src/data_processing/fetch_pdbs.py

```

## Step 3: Generate Data

Open the src/data_processing/generate_data.py file, and change the data_folder
variable to the path to the new dataset folder. Within the file there are also
other parameters which change be changed accordingly. The size variable
determines the width of the voxel space. The resolution variable determines the
resolution of one voxel in angstroms. The thresh variable determines how much of
a proteins stucture must be within the window in order to generate data for it.
The nb_rot variable determines the number of random rotation augmentations to
make for each protein. The channels variable determines the channels encoded
into data and are methods defined within the /src/data_processing/channels.py.

```
# Data folder path
data_folder = '../../data/my_dataset/'

# Data generator parameters
size = 64               # Voxel matrix size ex. 64 -> 64**3 space
resolution = 1.0        # Resolution of unit voxel
thresh = 0.95           # percentage of protein which must be inside window
nb_rot = 15              # Number of random rotation augmentations
channels = [aliphatic_res, aromatic_res, neutral_res, acidic_res, basic_res,
unique_res, alpha_carbons, beta_carbons]

################################################################################

```

Run the generate_data.py script and wait for all data to be generated. This script
has been parallelized using MPI.

```
$ python src/data_processing/generate_data.py

or

$ mpirun -n N python src/data_processing/generate_data.py

```

## Step 4: Generate HDF5 of Dataset

Open the src/data_processing/generate_hdf5.py and change the data_folder variable
to the path of the new dataset folder. The split variable determines the random split
of the data into training, validation and test respectively. The save_1d variable
determines whether to save the 1D version of the data into the final HDF5 file.
The save_2d variable determines whether to save the 2D version of the data into
the final HDF5 file. The save_3d variable determines whether to save the 3D
version of the data into the final HDF5 file. The nb_rot variable should be set to
the number of rotation augmentation generated.


```
# Data folder path
data_folder = '../../data/KrasHras/'

# Parameters
split = [0.7, 0.1, 0.2]
save_1d = True
save_2d = True
save_3d = True
nb_rot = 15

################################################################################

```

Run the generate_hdf5.py script and the dataset will be saved as dataset.hdf5 within
the new dataset folder.
