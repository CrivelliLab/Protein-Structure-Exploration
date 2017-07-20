'''
MeasurePDBs.py
Updated: 7/20/17
[NOT PASSING]

README:

The following script is used to calculate the 3D size distribution of processed
PDB data.

Global variables used to encode are defined under #- Global Variables.
'processed_file' defines the array file containing processed pdb data and rotation permutations.
Array file must be under data/interim/.

Command Line Interface:

$ python PDBWindower.py [-h] processed_file

'''
import os, argparse
from time import time
import numpy as np
from scipy import stats

#- Global Variables
processed_file = ''
bounds = None

#- debug Settings
debug = True
processed_file_usage = "processed PDB .npy file"
bounds_usage = "save PDB id list of PDBs that fit within define bounds;comma seperated"

################################################################################

def apply_rotation(pdb_data, rotation, debug=False):
    '''
    Method applies rotation to pdb_data defined as list of rotation matricies.

    '''
    if debug: print "Applying Rotation..."; t = time()

    rotated_pdb_data = []
    for i in range(len(pdb_data)):
        channel = []
        for coord in pdb_data[i]:
            temp = np.dot(rotation, coord[1:])
            temp = [coord[0], temp[0], temp[1], temp[2]]
            channel.append(np.array(temp))
        rotated_pdb_data.append(np.array(channel))
    rotated_pdb_data = np.array(rotated_pdb_data)

    if debug: print time() - t, 'secs...'

    return rotated_pdb_data

if __name__ == '__main__':

    # Cmd Line Args
    parser = argparse.ArgumentParser()
    parser.add_argument('processed_file', help=processed_file_usage, type=str)
    parser.add_argument('-b','--bounds', help=bounds_usage, type=str, default=None)
    args = vars(parser.parse_args())
    processed_file = args['processed_file']
    if args['bounds']: bounds = [float(x) for x in args['bounds'].split(',')]

    # File Paths
    os.chdir(os.path.dirname(os.path.realpath(__file__)))
    processed_file = '../../data/interim/' + processed_file

    # Load Data
    if debug: print("Loading PDB Data and Rotations...")
    data = np.load(processed_file)
    pdbs_data = data[0]

    # Process Rotations
    if debug: print("Measuring PDB diameters...")
    diameters = []
    for i in range(len(pdbs_data)):
        # Calculate Diameter
        dia = 0
        pdb_data  = pdbs_data[i][1]
        for channel in pdb_data:
            for atom in channel:
                dist = np.linalg.norm(atom[1:])
                if dist > dia: dia = dist
        dia *= 2
        diameters.append(np.round(dia))
    diameters = np.array(diameters)

    # Print Stats
    print "Diameter Stats of:", processed_file[6:]
    print 'Total', len(diameters)
    print "Mean", np.mean(diameters)
    print "Mode", stats.mode(diameters)[0][0]
    print "Max", np.max(diameters)
    print "Min", np.min(diameters)
    print "Std", np.std(diameters)

    if bounds:
        file_path = '../../data/raw/PDB/'+processed_file.split('_')[0].split('/')[-1]
        file_path += 'BOUNDED'+str(int(bounds[0]))+"%"+str(int(bounds[1]))+'.csv'
        with open(file_path, 'w') as f:
            i = 0
            for j in range(len(diameters)):
                if diameters[j] > bounds[0] and diameters[j] < bounds[1]:
                    pdb_id = pdbs_data[j][0][:4]
                    pdb_chain = pdbs_data[j][0][4]
                    f.write(pdb_id+','+pdb_chain+'\n')
                    i +=1
        print "PDBs within ", bounds[0], 'and', bounds[1], ':', i
        print "PDB ids saved in:", file_path[6:]
