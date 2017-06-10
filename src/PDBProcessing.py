'''
PDBProcessing.py
Last Updated: 5/9/2017

This script is used to parse and process Protein Data Base entries.

'''
import os
import numpy as np
import scipy.misc

# PDB File Parsing
from prody import *
confProDy(verbosity='none')

# Space Filling Curves and Visualization Tools
from HilbertCurves import *
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

    # Set Protein's Center Of Mass At Origin
    moveAtoms(protein, to=np.zeros(3))

    # Gather Atom Information
    atoms_residual = np.expand_dims(protein.getResnames(), 1)
    atoms_element = np.expand_dims(protein.getElements(), 1)
    atoms_radii = np.expand_dims([elem_radii[k] for k in protein.getElements()], 1)
    atoms_coords = protein.getCoords()
    pdb_data = np.concatenate([atoms_element, atoms_residual, atoms_radii, atoms_coords], 1)

    return pdb_data

def gen_3d_pdb(pdb_data):
    '''
    Method processes PDB's atom data to create a matrix based 3d model.

    '''
    # Coordinate, Radius Information
    xx = pdb_data[:,3].astype('float')
    yy = pdb_data[:,4].astype('float')
    zz = pdb_data[:,5].astype('float')
    ss = pdb_data[:,2].astype('float')

    # Generate Mesh For Protein
    if debug: print("Generating Mesh...")
    append_filter = vtk.vtkAppendPolyData()
    for i in range(len(pdb_data)):
        input1 = vtk.vtkPolyData()
        sphere_source = vtk.vtkSphereSource()
        sphere_source.SetCenter(xx[i],yy[i],zz[i])
        sphere_source.SetRadius(ss[i])
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
    voxel_modeller.SetSampleDimensions(64,64,64)
    voxel_modeller.SetModelBounds(-50,50,-50,50,-50,50)
    voxel_modeller.SetMaximumDistance(0.1)
    voxel_modeller.SetScalarTypeToInt()
    voxel_modeller.Update()
    voxel_array = vtk.util.numpy_support.vtk_to_numpy(voxel_modeller.GetOutput().GetPointData().GetScalars())
    voxel_array = voxel_array.reshape((64, 64, 64))

    return voxel_array

def encode_3d_pdb(pdb_3d, curve_3d, curve_2d):
    '''
    Method processes 3D PDB model and encodes into 2D image.

    '''
    # Dimension Reduction Using Space Filling Curve to 1D
    pdb_1d = np.zeros([len(curve_3d),])
    for i in range(len(curve_3d)):
        pdb_1d[i] = pdb_3d[curve_3d[i][0], curve_3d[i][1], curve_3d[i][2]]

    # Dimension Reconstruction Using Space Filling Curve to 2D
    s = int(np.sqrt(len(curve_2d)))
    pdb_2d = np.zeros([s,s])
    for i in range(len(pdb_1d)):
        pdb_2d[curve_2d[i][0], curve_2d[i][1]] = pdb_1d[i]

    if visualize:
        display_3d_array(pdb_3d)
        display_2d_array(pdb_2d)

if __name__ == '__main__':
    # Generate Hilbert Curves
    print "Generating Curves..."
    hilbert_3d = gen_hilbert_3D(6)
    hilbert_2d = gen_hilbert_2D(9)
    print "Generating Curves Done."

    # Read Ras PDBs
    pdb_files = []
    for line in sorted(os.listdir('../data/Ras-Gene-PDB-Files')): pdb_files.append(line)
    if debug: print "Total PDB Entries:", len(pdb_files)

    # Process PDB Entries
    for pdb_file in pdb_files:
        pdb_data = get_pdb_data('../data/Ras-Gene-PDB-Files/'+ pdb_file)
        pdb_3d_model = gen_3d_pdb(pdb_data)
        encoded_pdb_2d = encode_3d_pdb(pdb_3d_model, hilbert_3d, hilbert_2d)
        #scipy.misc.imsave(pdb_file[:-7]+'.png', encoded_pdb_2d)
