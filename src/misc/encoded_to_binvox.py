'''
png_to_binvox.pycd
Updated: 9/7/17

TODO:

README:

'''
import os
import numpy as np
from binvox import write_binvox
from scipy import misc
from mpi4py import MPI

data_folder = '../../data/raw/NEW_KRAS_HRAS/'
binvoxed_folder = '../../data/raw/NEW_KRAS_HRAS_BINVOX/'
curve_3d = 'hilbert_3d_6.npy'
curve_2d = 'hilbert_2d_9.npy'

################################################################################

def map_2d_to_3d(array_2d, curve_3d, curve_2d):
    '''
    Method maps 3D PDB array into 2D array.

    '''
    # Dimension Reduction Using Space Filling Curves from 3D to 2D
    s = int(np.cbrt(len(curve_3d)))
    array_3d = np.zeros([s,s,s])
    for i in range(len(curve_3d)):
        c2d = curve_2d[i]
        c3d = curve_3d[i]
        array_3d[c3d[0], c3d[1], c3d[2]] = array_2d[c2d[1], c2d[0]]

    return array_3d

if __name__ == '__main__':

    # File Paths
    os.chdir(os.path.dirname(os.path.realpath(__file__)))
    curve_2d = '../../data/misc/'+ curve_2d
    curve_3d = '../../data/misc/'+ curve_3d

    # Load Curves
    curve_3d = np.load(curve_3d)
    curve_2d = np.load(curve_2d)

    # MPI Init
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    cores = comm.Get_size()

    if rank == 0:
        if not os.path.exists(binvoxed_folder): os.makedirs(binvoxed_folder)
        entries = []
        for folder in sorted(os.listdir(data_folder)):
            if not os.path.isdir(data_folder+folder): continue
            if not os.path.exists(binvoxed_folder+folder):
                os.makedirs(binvoxed_folder+folder)
            for file_ in sorted(os.listdir(data_folder+folder)):
                if not file_.endswith('.png'): continue
                path = data_folder+folder +'/'+file_
                binvoxed_path = binvoxed_folder+folder+'/'+file_[:-4]+'.binvox'
                entries.append([path, binvoxed_path])

        entires = np.array(entries)
        np.random.shuffle(entries)
    else: entries = None
    entries = comm.bcast(entries, root=0)
    entries = np.array_split(entries, cores)[rank]
    print(len(entries))

    # Convert PNG Data to Binvox
    for i in range(len(entries)):
        path = entries[i][0]
        binvoxed_path = entries[i][1]
        img = misc.imread(path)/255.0
        img = img.astype(np.int)
        model = []
        for i in range(img.shape[2]):
            chan_2d = img[:,:,i]
            chan_3d = map_2d_to_3d(chan_2d, curve_3d, curve_2d)
            model.append(chan_3d)
        model = np.array(model).astype(np.bool)
        write_binvox(binvoxed_path, model)
