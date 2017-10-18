'''
PDB_DataGenerator.py
Updated: 10/6/17

'''
import os
import numpy as np
from itertools import product
from time import time
from scipy import misc

################################################################################

class PDB_DataGenerator(object):
    """
    Class creates a data generator which takes raw PDBs and creates volumetric
    representations of the atomic data.

    """
    # Hard Coded Values

    van_der_waal_radii = {  'H' : 1.2, 'C' : 1.7, 'N' : 1.55, 'O' : 1.52, 'S' : 1.8,
    'D' : 1.2, 'F' : 1.47, 'CL' : 1.75, 'BR' : 1.85, 'P' : 1.8,
    'I' : 1.98, '' : 0} # Source:https://physlab.lums.edu.pk/images/f/f6/Franck_ref2.pdf
    max_radius = 2.0

    def __init__(self, size=64, center=[0,0,0], resolution=1.0, nb_rotations=0, seed=9999):
        '''
        '''
        self.size = size
        self.center = center
        self.resolution = resolution
        self.bounds = [ -(size * resolution)/2, -(size * resolution)/2,
                        -(size * resolution)/2, (size * resolution)/2,
                        (size * resolution)/2,  (size * resolution)/2]

        self.tolerance = int(self.max_radius/resolution)
        self.tolerance_perms = np.array([x for x in product(*[[z for z in
                range(-self.tolerance, self.tolerance+1)] for y in range(3)])])

    def generate(self):
        '''
        '''
        pass

    def parse_pdb(self, path):
        '''
        Method parses atomic coordinate data from PDB. Coordinates are center
        around the centroid of the protein and then translated to the center
        coordinate defined for the DataGenerator.

        '''
        # Get Chain Information
        chain = path.split('/')[-1].split('.')[0].split('_')[1]

        # Parse Coordinates
        data = []
        with open(path, 'r') as f:
            lines = f.readlines()
            for row in lines:
                cols = row.split()
                if cols[0] == 'ATOM' and cols[4] == chain:
                    parsed_data = [cols[3], cols[2], self.van_der_waal_radii[cols[11]], cols[6], cols[7], cols[8]]
                    data.append(parsed_data)

        # Center Coordinates Around Centroid
        data = np.array(data)
        coords = data[:,3:].astype('float')
        centroid = np.mean(coords, axis=0)
        centered_coord = coords - centroid - self.center
        data = np.concatenate([data[:,:3], centered_coord], axis=1)

        return data

    def remove_outlier_atoms(self, data):
        '''
        Method removes atoms outside of the window defined by the size of the
        voxel map, center of window and resolution of window.

        '''
        # Get Atomic Coordinates
        coords = data[:,3:].astype('float')

        # Get Indexes of Outlier Atoms
        i = np.concatenate([np.where(coords[:,0] < self.bounds[0] - self.tolerance)[0],
                            np.where(coords[:,1] < self.bounds[1] - self.tolerance)[0],
                            np.where(coords[:,2] < self.bounds[2] - self.tolerance)[0],
                            np.where(coords[:,0] > self.bounds[3] + self.tolerance)[0],
                            np.where(coords[:,1] > self.bounds[4] + self.tolerance)[0],
                            np.where(coords[:,2] > self.bounds[5] + self.tolerance)[0]], axis=0)

        # Delete Outliers
        data = np.delete(data, i, axis=0)

        return data

    def calc_distances_from_voxels(self, data):
        '''
        Method calculates the distances from atoms to voxel centers for all atoms and
        voxels within the window.

        '''
        # Calculate Distances
        coords = data[:,3:].astype('float')
        nearest_voxels = np.rint(((coords - self.bounds[:3] + (self.resolution/2.0)) / self.resolution) - 1).astype('int')
        nearest_voxels_repeat = np.repeat(nearest_voxels, len(self.tolerance_perms), axis=0)
        tolerance_perms_repeat = np.tile(self.tolerance_perms, (len(nearest_voxels),1))
        nearest_voxels_with_tolerance = nearest_voxels_repeat + tolerance_perms_repeat
        nearest_voxels_coords = ((nearest_voxels_with_tolerance + (self.resolution/2.0)) * self.resolution) + self.bounds[:3]
        coords_repeat = np.repeat(coords, len(self.tolerance_perms), axis=0)
        distances = np.linalg.norm(coords_repeat -  nearest_voxels_coords, axis=1)
        data = np.repeat(data, len(self.tolerance_perms), axis=0)

        # Get Outlier Indexes
        i = np.concatenate([np.where(np.min(nearest_voxels_with_tolerance, axis=1) < 0)[0],
            np.where(np.max(nearest_voxels_with_tolerance, axis=1) > self.size-1)[0]], axis=0)

        # Delete outlier indexes
        data = np.delete(data, i, axis=0)
        distances = np.delete(distances, i, axis=0)
        nearest_voxels_with_tolerance = np.delete(nearest_voxels_with_tolerance, i, axis=0)

        return data, distances, nearest_voxels_with_tolerance

    def apply_channels(self, data, distances, nearest_voxels_indexes):
        '''
        Method extracts channel information from data and returns voxel_source
        indexes with corresponding values.

        '''
        # Remove Voxels Outside Atom Radius
        i = np.where((distances-data[:,2].astype('float')) > 0)
        data = np.delete(data, i, axis=0)
        distances = np.delete(distances, i, axis=0)
        nearest_voxels_indexes = np.delete(nearest_voxels_indexes, i, axis=0)

        # Split Channels
        chans = np.zeros((len(nearest_voxels_indexes),1)).astype('int')
        channels = [hydrophobic_res, polar_res, charged_res, alpha_carbons, beta_carbons]
        for i in range(len(channels)):
            x = ['1'] + ['0' for j in range(i)]
            x = ''.join(x)
            indexes = channels[i](data)
            chans[indexes] += int(x, 2)

        # Get Occupancy Values and Voxel Indexes
        values = np.ones(len(data))
        #values = np.exp((-4*(np.square(distances)))/np.square(data[:,2].astype('float')))
        voxel_indexes = np.concatenate([nearest_voxels_indexes, chans], axis=1)

        return voxel_indexes, values

    def generate_voxel_rep(self, voxel_indexes, values):
        '''
        Method generates nxnxnxc voxel representations of the channel occupancies.

        '''
        vox_3d = np.zeros((len(list(bin(np.max(voxel_indexes[:,3]))[2:])), self.size, self.size, self.size))
        for i in range(len(voxel_indexes)):
            ind = voxel_indexes[i,:3]
            chans = list(bin(voxel_indexes[i,3])[2:])
            for j in range(len(chans)):
                z = int(chans[j])
                if z == 1: vox_3d[len(chans)-1 - j, ind[0],ind[1],ind[2]] = values[i]
        return vox_3d

    def map_3d_to_2d(self, array_3d, curve_3d, curve_2d):
        '''
        Method maps 3D PDB array into 2D array.

        '''

        # Dimension Reduction Using Space Filling Curves from 3D to 2D
        s = int(np.sqrt(len(curve_2d)))
        array_2d = np.zeros([len(array_3d), s,s])
        for i in range(len(curve_3d)):
            c2d = curve_2d[i]
            c3d = curve_3d[i]
            for j in range(len(array_3d)):
                array_2d[j, c2d[0], c2d[1]] = array_3d[j, c3d[0], c3d[1], c3d[2]]

        return array_2d

    def hilbert_3d(order):
        '''
        Method generates 3D hilbert curve of desired order.

        Param:
            order - int ; order of curve

        Returns:
            np.array ; list of (x, y, z) coordinates of curve

        '''

        def gen_3d(order, x, y, z, xi, xj, xk, yi, yj, yk, zi, zj, zk, array):
            if order == 0:
                xx = x + (xi + yi + zi)/3
                yy = y + (xj + yj + zj)/3
                zz = z + (xk + yk + zk)/3
                array.append((xx, yy, zz))
            else:
                gen_3d(order-1, x, y, z, yi/2, yj/2, yk/2, zi/2, zj/2, zk/2, xi/2, xj/2, xk/2, array)

                gen_3d(order-1, x + xi/2, y + xj/2, z + xk/2,  zi/2, zj/2, zk/2, xi/2, xj/2, xk/2,
                           yi/2, yj/2, yk/2, array)
                gen_3d(order-1, x + xi/2 + yi/2, y + xj/2 + yj/2, z + xk/2 + yk/2, zi/2, zj/2, zk/2,
                           xi/2, xj/2, xk/2, yi/2, yj/2, yk/2, array)
                gen_3d(order-1, x + xi/2 + yi, y + xj/2+ yj, z + xk/2 + yk, -xi/2, -xj/2, -xk/2, -yi/2,
                           -yj/2, -yk/2, zi/2, zj/2, zk/2, array)
                gen_3d(order-1, x + xi/2 + yi + zi/2, y + xj/2 + yj + zj/2, z + xk/2 + yk +zk/2, -xi/2,
                           -xj/2, -xk/2, -yi/2, -yj/2, -yk/2, zi/2, zj/2, zk/2, array)
                gen_3d(order-1, x + xi/2 + yi + zi, y + xj/2 + yj + zj, z + xk/2 + yk + zk, -zi/2, -zj/2,
                           -zk/2, xi/2, xj/2, xk/2, -yi/2, -yj/2, -yk/2, array)
                gen_3d(order-1, x + xi/2 + yi/2 + zi, y + xj/2 + yj/2 + zj , z + xk/2 + yk/2 + zk, -zi/2,
                           -zj/2, -zk/2, xi/2, xj/2, xk/2, -yi/2, -yj/2, -yk/2, array)
                gen_3d(order-1, x + xi/2 + zi, y + xj/2 + zj, z + xk/2 + zk, yi/2, yj/2, yk/2, -zi/2, -zj/2,
                           -zk/2, -xi/2, -xj/2, -xk/2, array)

        n = pow(2, order)
        hilbert_curve = []
        gen_3d(order, 0, 0, 0, n, 0, 0, 0, n, 0, 0, 0, n, hilbert_curve)

        return np.array(hilbert_curve)

    def hilbert_2d(order):
        '''
        Method generates 2D hilbert curve of desired order.

        Param:
            order - int ; order of curve

        Returns:
            np.array ; list of (x, y) coordinates of curve

        '''
        def gen_2d(order, x, y, xi, xj, yi, yj, array):
            if order == 0:
                xx = x + (xi + yi)/2
                yy = y + (xj + yj)/2
                array.append((xx, yy))
            else:
                gen_2d(order-1, x, y, yi/2, yj/2, xi/2, xj/2, array)
                gen_2d(order-1, x + xi/2, y + xj/2, xi/2, xj/2, yi/2, yj/2, array)
                gen_2d(order-1, x + xi/2 + yi/2, y + xj/2 + yj/2, xi/2, xj/2, yi/2, yj/2, array)
                gen_2d(order-1, x + xi/2 + yi, y + xj/2 + yj, -yi/2,-yj/2,-xi/2,-xj/2, array)

        n = pow(2, order)
        hilbert_curve = []
        gen_2d(order, 0, 0, n, 0, 0, n, hilbert_curve)

        return np.array(hilbert_curve)

    def gen_random_rotations(nb_rot):
        '''
        Method generates random rotations by sampling hypersphere.

        For more information see this link, the last example of which inspired this
        approach:
            http://mathworld.wolfram.com/SpherePointPicking.html

        '''
        # For consistant random coordinate generation.
        np.random.seed(self.seed)

        # Sample rotations
        vector = np.random.randn(nb_rot, 3)
        vector /= np.linalg.norm(vector, axis=0)
        xi, yi, zi = vector
        print(vector.shape)

        # Combine the three Nx1 coordinate arrays into one Nx3 array describing x, y ,
        # z points.
        coordinate_arry = np.stack((xi, yi, zi), axis=-1)
        print coordinate_arry.shape

        # Convert to Rotation Matrix
        rotations = []
        for c in coordinate_arry:
            angle = np.arctan(c[2]/np.sqrt((c[0]**2)+(c[1]**2)))
            axis = np.dot(np.array([[0,1],[-1,0]]), np.array([c[0],c[1]]))
            rot1 = get_rotation_matrix([axis[0],axis[1],0],angle)
            if c[0] < 0 and c[1] < 0 or c[0] < 0 and c[1] > 0 :
                rot2 = get_rotation_matrix([0,0,1], np.arctan(c[1]/c[0]) + np.pi)
            else: rot2 = get_rotation_matrix([0,0,1], np.arctan(c[1]/c[0]))
            rot = np.dot(rot1, rot2)
            rotations.append(rot)
        rotations = np.array(rotations)

        return rotations

    def apply_rotation(pdb_data, rotation):
        '''
        Method applies rotation to pdb_data defined as list of rotation matricies.

        '''
        rotated_pdb_data = []
        for i in range(len(pdb_data)):
            channel = []
            for coord in pdb_data[i]:
                temp = np.dot(rotation, coord[1:])
                temp = [coord[0], temp[0], temp[1], temp[2]]
                channel.append(np.array(temp))
            rotated_pdb_data.append(np.array(channel))
        rotated_pdb_data = np.array(rotated_pdb_data)

        return rotated_pdb_data

def hydrophobic_res(data):
    '''
    '''
    # Hydrophobic Residues
    i = np.concatenate([np.where(data[:,0] == 'ALA')[0],
                        np.where(data[:,0] == 'ILE')[0],
                        np.where(data[:,0] == 'LEU')[0],
                        np.where(data[:,0] == 'MET')[0],
                        np.where(data[:,0] == 'PHE')[0],
                        np.where(data[:,0] == 'PRO')[0]], axis=0)
    return i

def polar_res(data):
    '''
    '''
    # Polar Residues
    i = np.concatenate([np.where(data[:,0] == 'ARG')[0], np.where(data[:,0] == 'ASN')[0],
                        np.where(data[:,0] == 'ASP')[0], np.where(data[:,0] == 'ASX')[0],
                        np.where(data[:,0] == 'CSO')[0], np.where(data[:,0] == 'CYS')[0],
                        np.where(data[:,0] == 'CYX')[0], np.where(data[:,0] == 'GLN')[0],
                        np.where(data[:,0] == 'GLU')[0], np.where(data[:,0] == 'GLX')[0],
                        np.where(data[:,0] == 'GLY')[0], np.where(data[:,0] == 'HID')[0],
                        np.where(data[:,0] == 'IE')[0], np.where(data[:,0] == 'HIP')[0],
                        np.where(data[:,0] == 'HIS')[0], np.where(data[:,0] == 'HSD')[0],
                        np.where(data[:,0] == 'HSE')[0], np.where(data[:,0] == 'HSP')[0],
                        np.where(data[:,0] == 'LYS')[0], np.where(data[:,0] == 'PTR')[0],
                        np.where(data[:,0] == 'SEP')[0], np.where(data[:,0] == 'SER')[0],
                        np.where(data[:,0] == 'THR')[0], np.where(data[:,0] == 'TPO')[0],
                        np.where(data[:,0] == 'YR')[0]], axis=0)
    return i

def charged_res(data):
    '''
    '''
    # Charged Residues
    i = np.concatenate([np.where(data[:,0] == 'ARG')[0],
                        np.where(data[:,0] == 'ASP')[0],
                        np.where(data[:,0] == 'GLU')[0],
                        np.where(data[:,0] == 'HIS')[0],
                        np.where(data[:,0] == 'LYS')[0]], axis=0)
    return i

def alpha_carbons(data):
    '''
    '''
    i = np.where(data[:,1] == 'CA')
    return i

def beta_carbons(data):
    '''
    '''
    i = np.where(data[:,1] == 'CB')
    return i
