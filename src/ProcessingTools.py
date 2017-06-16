'''
ProcessingTools.py
Last Updated: 6/16/2017

'''
import os
import numpy as np

# PDB File Parsing
from prody import *
confProDy(verbosity='none')

# 3D Modeling and Rendering
import vtk
import vtk.util.numpy_support as vtk_np
from scipy import ndimage

# Hard Coded Knowledge
residuals = [   'ALA', 'ARG', 'ASN', 'ASP', 'ASX', 'CYS', 'GLN', 'GLU', 'GLX',
                'GLY', 'HIS', 'ILE', 'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER',
                'THR', 'TRP', 'TYR', 'UNK', 'VAL']
elem_radii = {  'H' : 1.2, 'C' : 1.7, 'N' : 1.55, 'O' : 1.52, 'S' : 1.8,
                'D' : 1.2, 'F' : 1.47, 'CL' : 1.75, 'BR' : 1.85, 'P' : 1.8,
                'I' : 1.98, '' : 0}


def get_pdb_data(pdb_file, channels=[], rot=None, debug=False):
    '''
    Method parses residual, element, radii and coordinate information for each
    atom present in the PDB file, and returns as numpy array.

    '''
    # Parse PDB File
    if debug: print "Parsing:", pdb_file
    molecule = parsePDB(pdb_file)
    molecule = molecule.select('protein')

    # Set Protein's Center Of Mas At Origin
    moveAtoms(molecule, to=np.zeros(3))

    # Gather Atom Information
    atoms_coords = molecule.getCoords()
    dia = np.max(np.absolute(atoms_coords))
    pdb_data = []
    for channel in channels:
        channel_ = molecule.select(channel)
        if channel_ is not None:
            channel_radii = [elem_radii[k] for k in channel_.getElements()]
            channel_radii = np.expand_dims(channel_radii, 1)
            channel_coords = channel_.getCoords()

            # Apply Rotation
            if rot is not None:
                for r in rot:
                    temp_coords = []
                    for coord in channel_coords:
                        temp_coords.append(np.dot(r, coord))
                    channel_coords = np.array(temp_coords)

            channel_data = np.concatenate([channel_radii, channel_coords], 1)
        else: channel_data = None
        pdb_data.append(channel_data)

    if debug: print "Minimum Diameter:", dia*2

    return pdb_data, dia

def gen_3d_stl(stl_file, rot, bounds, sample_dim, debug=False):
    '''
    '''
    # Generate Mesh For Protein
    if debug: print("Generating Mesh...")
    reader = vtk.vtkSTLReader()
    reader.SetFileName(stl_file)
    reader.Update()

    # Voxelize Mesh
    if debug: print("Voxelizing Mesh...")
    voxel_modeller = vtk.vtkVoxelModeller()
    voxel_modeller.SetInputConnection(reader.GetOutputPort())
    voxel_modeller.SetSampleDimensions(sample_dim, sample_dim, sample_dim)
    if bounds is None:
        bounds = reader.GetOutput().GetBounds()
        x_range = bounds[1] - bounds[0]
        y_range = bounds[3] - bounds[2]
        z_range = bounds[5] - bounds[4]
        max_rad = max([x_range, y_range, z_range])/2
        temp = []
        temp.append(bounds[0]+(x_range/2)-max_rad)
        temp.append(bounds[1]-(x_range/2)+max_rad)
        temp.append(bounds[2]+(y_range/2)-max_rad)
        temp.append(bounds[3]-(y_range/2)+max_rad)
        temp.append(bounds[4]+(z_range/2)-max_rad)
        temp.append(bounds[5]-(z_range/2)+max_rad)
        bounds = temp
    x0, x1, y0, y1, z0, z1 = bounds
    voxel_modeller.SetModelBounds(x0, x1, y0, y1, z0, z1)
    voxel_modeller.SetMaximumDistance(0.01)
    voxel_modeller.SetScalarTypeToInt()
    voxel_modeller.Update()
    voxel_output = voxel_modeller.GetOutput().GetPointData().GetScalars()
    voxel_array = vtk_np.vtk_to_numpy(voxel_output)
    voxel_array = voxel_array.reshape((sample_dim, sample_dim, sample_dim))

    # Fill Interiors
    if debug: print("Filling Interiors...")
    filled_voxel_array = []
    for sect in voxel_array:
        filled_sect = ndimage.morphology.binary_fill_holes(sect).astype('int')
        filled_voxel_array.append(filled_sect)
    filled_voxel_array = np.array(filled_voxel_array)

    return filled_voxel_array

def gen_3d_pdb(pdb_data, bounds, sample_dim, debug=False):
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
    voxel_modeller.SetMaximumDistance(0.01)
    voxel_modeller.SetScalarTypeToInt()
    voxel_modeller.Update()
    voxel_output = voxel_modeller.GetOutput().GetPointData().GetScalars()
    voxel_array = vtk_np.vtk_to_numpy(voxel_output)
    voxel_array = voxel_array.reshape((sample_dim, sample_dim, sample_dim))

    # Fill Interiors
    if debug: print("Filling Interiors...")
    filled_voxel_array = []
    for sect in voxel_array:
        filled_sect = ndimage.morphology.binary_fill_holes(sect).astype('int')
        filled_voxel_array.append(filled_sect)
    filled_voxel_array = np.array(filled_voxel_array)

    return filled_voxel_array

def encode_3d_pdb(pdb_3d, curve_3d, curve_2d, debug=False):
    '''
    Method proceses 3D PDB model and encodes into 2D image.

    '''
    if debug: print('Applying 3D to 1D Space Filling Curve...')

    # Dimension Reduction Using Space Filling Curve to 1D
    pdb_1d = np.zeros([len(curve_3d),])
    for i in range(len(curve_3d)):
        pdb_1d[i] = pdb_3d[curve_3d[i][0], curve_3d[i][1], curve_3d[i][2]]

    if debug: print('Applying 1D to 2D Space Filling Curve...')

    # Dimension Recasting Using Space Filling Curve to 2D
    s = int(np.sqrt(len(curve_2d)))
    pdb_2d = np.zeros([s,s])
    for i in range(len(pdb_1d)):
        pdb_2d[curve_2d[i][0], curve_2d[i][1]] = pdb_1d[i]

    return pdb_2d

def rotation_matrix(axis, theta):
    '''
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians.
    '''
    axis = np.asarray(axis)
    axis = axis/np.sqrt(np.dot(axis, axis))
    a = np.cos(theta/2.0)
    b, c, d = -axis*np.sin(theta/2.0)
    aa, bb, cc, dd = a*a, b*b, c*c, d*d
    bc, ad, ac, ab, bd, cd = b*c, a*d, a*c, a*b, b*d, c*d
    return np.array([[aa+bb-cc-dd, 2*(bc+ad), 2*(bd-ac)],
                     [2*(bc-ad), aa+cc-bb-dd, 2*(cd+ab)],
                     [2*(bd+ac), 2*(cd-ab), aa+dd-bb-cc]])
