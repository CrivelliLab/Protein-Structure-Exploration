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

# Global Variables
debug = True
visualize = True
residuals = [   'ALA', 'ARG', 'ASN', 'ASP', 'ASX', 'CYS', 'GLN', 'GLU', 'GLX',
                'GLY', 'HIS', 'ILE', 'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER',
                'THR', 'TRP', 'TYR', 'UNK', 'VAL']
elem_radii = {  'H' : 1.2, 'C' : 1.7, 'N' : 1.55, 'O' : 1.52, 'S' : 1.8, 'D' : 1.2}

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
    atoms_residual = np.expand_dims(protein.getResnames(), 1)
    atoms_element = np.expand_dims(protein.getElements(), 1)
    atoms_radii = np.expand_dims([elem_radii[k] for k in protein.getElements()], 1)
    atoms_coords = protein.getCoords()
    pdb_data = np.concatenate([atoms_element, atoms_residual, atoms_radii, atoms_coords], 1)

    if debug: print "Max Dimension:", np.max(np.absolute(atoms_coords))

    return pdb_data

def gen_3d_pdb(pdb_data, bounds, sample_dim):
    '''
    Method proceses PDB's atom data to create a matrix based 3d model.

    '''
    # Coordinate, Radius Information
    x = pdb_data[:,3].astype('float')
    y = pdb_data[:,4].astype('float')
    z = pdb_data[:,5].astype('float')
    s = pdb_data[:,2].astype('float')

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
    voxel_modeller.SetMaximumDistance((x1-x0)/sample_dim)
    voxel_modeller.SetScalarTypeToInt()
    voxel_modeller.Update()
    voxel_array = vtk.util.numpy_support.vtk_to_numpy(voxel_modeller.GetOutput().GetPointData().GetScalars())
    voxel_array = voxel_array.reshape((sample_dim, sample_dim, sample_dim))

    if visualize: display_3d_array(voxel_array)

    return voxel_array

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

    if visualize: display_2d_array(pdb_2d)

if __name__ == '__main__':
    # Generate Hilbert Curves
    print "Generating Curves..."
    zcurve_3d = gen_zcurve_3D(pow(256, 3))
    zcurve_2d = gen_zcurve_2D(pow(256, 3))
    print "Generating Curves Done."

    # Read Ras PDBs
    pdb_files = []
    for line in sorted(os.listdir('../data/Ras-Gene-PDB-Files')): pdb_files.append(line)
    if debug: print "Total PDB Entries:", len(pdb_files)

    # Proces PDB Entries
    for pdb_file in pdb_files:
        pdb_data = get_pdb_data('../data/Ras-Gene-PDB-Files/'+ pdb_file)
        pdb_3d_model = gen_3d_pdb(pdb_data, (-25, 25, -25, 25, -25, 25), 256)
        encoded_pdb_2d = encode_3d_pdb(pdb_3d_model, zcurve_3d, zcurve_2d)
