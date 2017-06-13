'''
PDBProcesing.py
Last Updated: 5/11/2017

This script is used to parse and proces Protein Data Base entries.

'''
import os
import numpy as np

# PDB File Parsing
from prody import *
confProDy(verbosity='none')

# Space Filling Curves
from HilbertCurves import *
from ZCurve import *

# Visualization Tools
from VisualizationTools import *

# 3D Modeling and Rendering
import vtk
from scipy import ndimage, misc

# Global Variables
div = 64
min_ = -70
max_ = 70
debug = True
visualize = False
stats = False
residuals = [   'ALA', 'ARG', 'ASN', 'ASP', 'ASX', 'CYS', 'GLN', 'GLU', 'GLX',
                'GLY', 'HIS', 'ILE', 'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER',
                'THR', 'TRP', 'TYR', 'UNK', 'VAL']
elem_radii = {  'H' : 1.2, 'C' : 1.7, 'N' : 1.55, 'O' : 1.52, 'S' : 1.8, 'D' : 1.2}

def run_stats(pdb_files):
    '''
    '''
    if debug: print("Running Stats...")
    min_diameters = []
    for pdb_file in pdb_files:
        # Parse PDB File
        protein = parsePDB('../data/Ras-Gene-PDB-Files/'+ pdb_file).select('protein')

        # Set Protein's Center Of Mas At Origin
        moveAtoms(protein, to=np.zeros(3))

        # Gather Atom Information
        atoms_coords = protein.getCoords()
        min_diameters.append(np.max(np.absolute(atoms_coords))*2)

    display_min_diameter_dist(min_diameters)

def get_pdb_data(pdb_file):
    '''
    Method parses residual, element, radii and coordinate information for each atom
    present in the PDB file, and returns as numpy array.

    '''
    # Parse PDB File
    if debug: print "Parsing:", pdb_file
    protein = parsePDB(pdb_file).select('protein')

    # Set Protein's Center Of Mas At Origin
    moveAtoms(protein, to=np.zeros(3))

    # Gather Atom Information
    #atoms_residual = np.expand_dims(protein.getResnames(), 1)
    #atoms_element = np.expand_dims(protein.getElements(), 1)
    #atoms_radii = np.expand_dims([elem_radii[k] for k in protein.getElements()], 1)
    atoms_coords = protein.getCoords()
    #pdb_data = np.concatenate([atoms_element, atoms_residual, atoms_radii, atoms_coords], 1)

    # Gather Hydrophobic Atom Information
    hydrophobic = protein.select('hydrophobic')
    hp_radii = np.expand_dims([elem_radii[k] for k in hydrophobic.getElements()], 1)
    hp_coords = hydrophobic.getCoords()
    hp_data = np.concatenate([hp_radii, hp_coords], 1)

    # Gather Polar Atom Information
    polar = protein.select('polar')
    p_radii = np.expand_dims([elem_radii[k] for k in polar.getElements()], 1)
    p_coords = polar.getCoords()
    p_data = np.concatenate([p_radii, p_coords], 1)

    # Gather Hydrophobic Atom Information
    charged = protein.select('charged')
    c_radii = np.expand_dims([elem_radii[k] for k in charged.getElements()], 1)
    c_coords = charged.getCoords()
    c_data = np.concatenate([c_radii, c_coords], 1)

    pdb_data = [hp_data, p_data, c_data]

    dia = np.max(np.absolute(atoms_coords))

    if debug: print "Minimum Diameter:", dia

    return pdb_data, dia

def gen_3d_pdb(pdb_data, bounds, sample_dim):
    '''
    Method proceses PDB's atom data to create a matrix based 3d model.

    '''

    # Coordinate, Radius Information
    x = pdb_data[:,3].astype('float')
    y = pdb_data[:,2].astype('float')
    z = pdb_data[:,1].astype('float')
    s = pdb_data[:,0].astype('float')

    # Generate Mesh For Protein
    if debug: print("Generating Mesh...")
    append_filter = vtk.vtkAppendPolyData()
    for i in range(len(pdb_data)):
        input1 = vtk.vtkPolyData()
        sphere_source = vtk.vtkSphereSource()
        sphere_source.SetCenter(x[i],y[i],z[i])
        sphere_source.SetRadius(s[i])
        sphere_source.Update()
        input1.ShallowCopy(sphere_source.GetOutput())
        if vtk.VTK_MAJOR_VERSION <= 5:
            append_filter.AddInputConnection(input1.GetProducerPort())
        else: append_filter.AddInputData(input1)
    append_filter.Update()

    #  Remove Any Duplicate Points.
    clean_filter = vtk.vtkCleanPolyData()
    clean_filter.SetInputConnection(append_filter.GetOutputPort())
    clean_filter.Update()

    # Voxelize Mesh
    if debug: print("Voxelizing Mesh...")
    voxel_modeller = vtk.vtkVoxelModeller()
    voxel_modeller.SetInputConnection(clean_filter.GetOutputPort())
    voxel_modeller.SetSampleDimensions(sample_dim, sample_dim, sample_dim)
    x0, x1, y0, y1, z0, z1 = bounds
    voxel_modeller.SetModelBounds(x0, x1, y0, y1, z0, z1)
    voxel_modeller.SetMaximumDistance(0.1)
    voxel_modeller.SetScalarTypeToInt()
    voxel_modeller.Update()
    voxel_array = vtk.util.numpy_support.vtk_to_numpy(voxel_modeller.GetOutput().GetPointData().GetScalars())
    voxel_array = voxel_array.reshape((sample_dim, sample_dim, sample_dim))

    # Fill Interiors
    if debug: print("Filling Interiors...")
    filled_voxel_array = []
    for sect in voxel_array:
        filled_sect = ndimage.morphology.binary_fill_holes(sect).astype('int')
        filled_voxel_array.append(filled_sect)
    filled_voxel_array = np.array(filled_voxel_array)

    return filled_voxel_array

def encode_3d_pdb(pdb_3d, curve_3d, curve_2d):
    '''
    Method proceses 3D PDB model and encodes into 2D image.

    '''
    if debug: print('Applying 3D to 1D Space Filling Curve...')

    # Dimension Reduction Using Space Filling Curve to 1D
    pdb_1d = np.zeros([len(curve_3d),])
    for i in range(len(curve_3d)):
        pdb_1d[i] = pdb_3d[curve_3d[i][0], curve_3d[i][1], curve_3d[i][2]]

    if debug: print('Applying 1D to 2D Space Filling Curve...')

    # Dimension Reconstruction Using Space Filling Curve to 2D
    s = int(np.sqrt(len(curve_2d)))
    pdb_2d = np.zeros([s,s])
    for i in range(len(pdb_1d)):
        pdb_2d[curve_2d[i][0], curve_2d[i][1]] = pdb_1d[i]

    return pdb_2d

if __name__ == '__main__':

    # Read Ras PDBs
    pdb_files = []
    for line in sorted(os.listdir('../data/Ras-Gene-PDB-Files')): pdb_files.append(line)
    if debug: print "Total PDB Entries:", len(pdb_files)

    if stats: run_stats(pdb_files)

    # Generate Hilbert Curves
    if debug: print("Generating 3D Curve...")
    zcurve_3d = gen_zcurve_3D(pow(div, 3))
    if debug: print("Generating 2D Curve...")
    zcurve_2d = gen_zcurve_2D(pow(div, 3))

    # Proces PDB Entries
    for pdb_file in pdb_files:
        pdb_data, dia = get_pdb_data('../data/Ras-Gene-PDB-Files/'+ pdb_file)
        encoded_pdb_2d = []
        pdb_3d_model = []
        for i in range(3):
            if debug: print('Processing Channel ' + str(i) + '...')
            pdb_data_res = pdb_data[i]
            pdb_3d_res = gen_3d_pdb(pdb_data_res, (-dia, dia, -dia, dia, -dia, dia), div)
            pdb_3d_model.append(pdb_3d_res)
            encoded_res_2d = encode_3d_pdb(pdb_3d_res, zcurve_3d, zcurve_2d)
            encoded_pdb_2d.append(encoded_res_2d)
        encoded_pdb_2d =  np.transpose(np.array(encoded_pdb_2d), (2,1,0))

        if debug: print("Saving Encoded PDB...")
        #encoded_pdb_2d = ndimage.interpolation.zoom(encododed_pdb_2d, 0.125)
        misc.imsave('../data/Processed-Ras-Gene-PDB-Files/'+pdb_file[:-7]+'.png', encoded_pdb_2d)

        if visualize:
            print('Visualizing All Channels...')
            display_3d_array(pdb_3d_model)
            display_2d_array(encoded_pdb_2d)
