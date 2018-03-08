'''
measure_diameters.py
Updated: 2/8/18

This script is used to measure the diameters of protein chains.

'''
import os
import numpy as np
from tqdm import tqdm
from sklearn.decomposition import PCA

# Parameters
data_folder = "../../data/T0882/"

################################################################################

def parse_pdb(path, chain, res_i, all_chains=False):
    '''
    Method parses atomic coordinate data from PDB. Coordinates are center
    around the centroid of the protein and then translated to the center
    coordinate defined for the DataGenerator.

    Params:
        path - str; PDB file path
        chain - str; chain identifier
        res_i - list[int]; list of residue indexes
        all_chains - boolean; whether all chains of PDB are used

    Returns:
        data - np.array; PDB atomic coordinate data

    '''

    van_der_waal_radii = {  'H' : 1.2, 'C' : 1.7, 'N' : 1.55, 'O' : 1.52,
    'S' : 1.8, 'D' : 1.2, 'F' : 1.47, 'CL' : 1.75, 'BR' : 1.85, 'P' : 1.8,
    'I' : 1.98, 'E' : 1.0, 'X':1.0 , '': 0.0}
    # Source:https://physlab.lums.edu.pk/images/f/f6/Franck_ref2.pdf

    # Parse Coordinates
    data = []
    with open(path, 'r') as f:
        lines = f.readlines()
        for row in lines:
            if row[:4] == 'ATOM' and row[21] == chain:
                if res_i != None:
                    if int(row[22:26]) in res_i:
                        parsed_data = [row[17:20], row[12:16].strip(), van_der_waal_radii[row[77].strip()], row[30:38], row[38:46], row[47:54]]
                else:
                    parsed_data = [row[17:20], row[12:16].strip(), van_der_waal_radii[row[77].strip()], row[30:38], row[38:46], row[47:54]]
                data.append(parsed_data)
            elif row[:4] == 'ATOM' and all_chains:
                if res_i != None:
                    if int(row[22:26]) in res_i:
                        parsed_data = [row[17:20], row[12:16].strip(), van_der_waal_radii[row[77].strip()], row[30:38], row[38:46], row[47:54]]
                else:
                    parsed_data = [row[17:20], row[12:16].strip(), van_der_waal_radii[row[77].strip()], row[30:38], row[38:46], row[47:54]]
                data.append(parsed_data)

    data = np.array(data)
    if len(data) == 0: return []

    # Center Coordinates Around Centroid
    coords = data[:,3:].astype('float')
    centroid = np.mean(coords, axis=0)
    centered_coord = coords - centroid

    # Orient along prime axis
    pca = PCA(n_components=3)
    pca.fit(centered_coord)
    c = pca.components_[np.argmax(pca.explained_variance_)]
    angle = np.arctan(c[2]/np.sqrt((c[0]**2)+(c[1]**2)))
    axis = np.dot(np.array([[0,1],[-1,0]]), np.array([c[0],c[1]]))
    rot1 = get_rotation_matrix([axis[0],axis[1],0],angle)
    if c[0] < 0 and c[1] < 0 or c[0] < 0 and c[1] > 0 :
        rot2 = get_rotation_matrix([0,0,1], np.arctan(c[1]/c[0]) + np.pi)
    else: rot2 = get_rotation_matrix([0,0,1], np.arctan(c[1]/c[0]))
    rot = np.dot(rot1, rot2)
    centered_coord = np.dot(centered_coord, rot)

    data = np.concatenate([data[:,:3], centered_coord], axis=1)

    del centroid, centered_coord, coords

    return data

def get_rotation_matrix(axis, theta):
    '''
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians.

    Param:
        axis - list ; (x, y, z) axis coordinates
        theta - float ; angle of rotaion in radians

    Return:
        rotation_matrix - np.array

    '''
    axis = np.asarray(axis)
    axis = axis/np.sqrt(np.dot(axis, axis))
    a = np.cos(theta/2.0)
    b, c, d = -axis*np.sin(theta/2.0)
    aa, bb, cc, dd = a*a, b*b, c*c, d*d
    bc, ad, ac, ab, bd, cd = b*c, a*d, a*c, a*b, b*d, c*d

    rotation_matrix = np.array([[aa+bb-cc-dd, 2*(bc+ad), 2*(bd-ac)],
                                [2*(bc-ad), aa+cc-bb-dd, 2*(cd+ab)],
                                [2*(bd+ac), 2*(cd-ab), aa+dd-bb-cc]])

    return rotation_matrix

if __name__ == '__main__':

    # Set paths relative to this file
    os.chdir(os.path.dirname(os.path.realpath(__file__)))

    # Parse data and measure diameter
    diameters = []
    for pdb in tqdm(sorted(os.listdir(data_folder+'pdbs'))):
        data = parse_pdb(data_folder+'pdbs/'+pdb, 'A', None, True)
        coords = data[:,3:].astype('float')
        diameters.append((np.max(coords)*2))
    diameters = np.array(diameters)

    # Display statistics
    print(np.max(diameters))
    print(np.min(diameters))
    print(np.average(diameters))
