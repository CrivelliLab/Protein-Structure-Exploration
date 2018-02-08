'''
parse_casp.py
Updated: 1/29/17

'''
import os
from shutil import copyfile
import numpy as np

# Data folder path
data_folder = '../../data/T0882/'

# Parameters
bins = [0.375, 1.0] # Bin cut-offs in ascending order

###############################################################################

seed = 1234

if __name__ == '__main__':

    # Set paths relative to this file
    os.chdir(os.path.dirname(os.path.realpath(__file__)))

    # Gather ids and GDTTS scores from results .txt
    ids = []
    scores = []
    with open(data_folder+data_folder.split('/')[-2]+'.csv', 'r') as f:
        lines = f.readlines()
        for i in range(len(lines)):
            x = lines[i].split(',')
            if i > 1 and len(x) == 4:
                id_ = x[0]
                score = float(x[1])
                ids.append(id_)
                scores.append(score)
    print(len(ids))

    # Bin ids according to score as defined in bins parameter
    binned_ids = []
    for i in range(len(bins)):
        bin_ = bins[i]
        ids_pos = []
        ids_neg = []
        scores_neg = []
        for j in range(len(ids)):
            if scores[j] < bin_:
                ids_pos.append(ids[j])
            else:
                ids_neg.append(ids[j])
                scores_neg.append(scores[j])

        print(len(ids_pos))
        binned_ids.append(ids_pos)
        ids = ids_neg
        scores = scores_neg

    '''
    # Copy PDBs into folders
    for i in range(len(bins)):
        bin_ = bins[i]
        if not os.path.exists(data_folder+'<'+str(bin_)): os.mkdir(data_folder+'<'+str(bin_))
        for j in range(len(binned_ids[i])):
            id_ = binned_ids[i][j]
            if not os.path.exists(data_folder+'<'+str(bin_)+'/'+id_): os.mkdir(data_folder+'<'+str(bin_)+'/'+id_)
            copyfile(data_folder+'/pdbs/'+id_+'.pdb', data_folder+'<'+str(bin_)+'/'+id_+'/'+id_+'.pdb')
    '''
