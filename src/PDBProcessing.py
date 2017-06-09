'''
PDBProcessing.py
Last Updated: 5/9/2017

This script is used to parse and process Protein Data Base entries.

'''
import os
import numpy as np
from tqdm import tqdm

# PDB File Parsing
from prody import *
confProDy(verbosity='none')

# Space Filling Curves and Visualization Tools
from HilbertCurves import *
from VisualizationTools import *

# 3D Modeling and Rendering
import vtk
import mayavi.mlab as mlab
from tvtk.api import tvtk

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
    protein = parsePDB(pdb_file).select('protein').chain_A

    # Set Protein's Center Of Mass At Origin
    moveAtoms(protein, to=np.zeros(3))

    # Gather Atom Information
    atoms_residual = np.expand_dims(protein.getResnames(), 1)
    atoms_element = np.expand_dims(protein.getElements(), 1)
    atoms_radii = np.expand_dims([elem_radii[k] for k in protein.getElements()], 1)
    atoms_coords = protein.getCoords()
    pdb_data = np.concatenate([atoms_element, atoms_residual, atoms_radii, atoms_coords], 1)

    return pdb_data

def sphere_mesh_template():
    '''
    Method creates sphere mesh template.
    '''
    phi, theta = np.mgrid[0:np.pi:11j, 0:2*np.pi:11j]
    x = np.sin(phi) * np.cos(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(phi)
    return x, y, z

def gen_3d_model(pdb_data, max_, min_, res_):
    '''
    Method processes PDB's atom data to create a matrix based 3d model.

    '''
    x, y, z = sphere_mesh_template()

    # Coordinate, Radius and Coloring Information
    xx = pdb_data[:,3].astype('float')
    yy = pdb_data[:,4].astype('float')
    zz = pdb_data[:,5].astype('float')
    ss = pdb_data[:,2].astype('float')
    cc = []
    for i in range(len(pdb_data)):
        if pdb_data[i][0] == 'C': cc.append((1,0,0))
        elif pdb_data[i][0] == 'H': cc.append((1,1,0))
        elif pdb_data[i][0] == 'O': cc.append((0,1,0))
        elif pdb_data[i][0] == 'S': cc.append((0,0,1))
        else: cc.append((1,1,1))
    cc = np.array(cc)

    # Generate Mesh for Protein
    if debug: print("Creating Mesh...")
    pbar = tqdm(total=len(pdb_data))
    poly_data = []
    for i in range(len(pdb_data)):
        x_ = (x * ss[i]) + xx[i]
        y_ = (y * ss[i]) + yy[i]
        z_ = (z * ss[i]) + zz[i]
        atom = mlab.mesh(x_, y_, z_, color=(cc[i][0], cc[i][1], cc[i][2]))
        pbar.update(1)
        polydata = tvtk.to_vtk(atom.actor.actors[0].mapper.input)
        poly_data.append(polydata)
    mesh_appender = vtk.vtkAppendPolyData()
    for polydata in poly_data:
        if vtk.VTK_MAJOR_VERSION <= 5:
            mesh_appender.AddInputConnection(polydata.GetProducerPort())
        else: mesh_appender.AddInputData(polydata)

    mlab.clf()
    # Create and visualize the mesh
    mesh_appender.Update()

    #  Remove any duplicate points.
    cleanFilter = vtk.vtkCleanPolyData()
    cleanFilter.SetInputConnection(mesh_appender.GetOutputPort())
    cleanFilter.Update()

    # Create a mapper and actor
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(cleanFilter.GetOutputPort())
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)

    # Create a renderer, render window, and interactor
    renderer = vtk.vtkRenderer()
    renderWindow = vtk.vtkRenderWindow()
    renderWindow.AddRenderer(renderer)
    renderWindowInteractor = vtk.vtkRenderWindowInteractor()
    renderWindowInteractor.SetRenderWindow(renderWindow)

    # Add the actors to the scene
    renderer.AddActor(actor)
    renderer.SetBackground(.3, .2, .1) #  Background color dark red

    # Render and interact
    renderWindow.Render()
    renderWindowInteractor.Start()

    # Bin x, y, z Coordinates
    pdb_data = pdb_data[:,3:].astype('float')
    range_ = max_ - min_
    bins = [(i*res_) + min_ for i in range(int(range_/res_)+1)]
    x_binned = np.digitize(pdb_data[:, 0], bins) - 1
    y_binned = np.digitize(pdb_data[:, 1], bins) - 1
    z_binned = np.digitize(pdb_data[:, 2], bins) - 1
    indices = np.array([x_binned, y_binned, z_binned])
    indices = np.transpose(indices, (1, 0))

    # Get Unique Indices And Counts
    u_indices = {}
    for ind in indices:
        ind_ = tuple(ind.tolist())
        if ind_ in u_indices: u_indices[ind_] += 1
        else: u_indices[ind_] = 1

    # Generate 3D Array
    pdb_3d = np.zeros([int(range_/res_)+1 for i in range(3)])
    for ind in u_indices.keys(): pdb_3d[ind[0], ind[1], ind[2]] = u_indices[ind]

    return pdb_3d

def encode_3d_pdb(pdb_3d, curve_3d, curve_2d):
    '''
    Method processes 3D PDB model and encodes into 2D image.

    '''
    # Dimension Reduction Using Space Filling Curve to 1D
    array_1d = np.zeros([len(curve_3d),])
    for i in range(len(curve_3d)):
        array_1d[i] = pdb_3d[curve_3d[i][0], curve_3d[i][1], curve_3d[i][2]]

    # Dimension Reconstruction Using Space Filling Curve to 2D
    s = int(np.sqrt(len(curve_2d)))
    array_2d = np.zeros([s,s])
    for i in range(len(array_1d)):
        array_2d[curve_2d[i][0], curve_2d[i][1]] = array_1d[i]

    if visualize:
        display_2d_array(array_2d)
        display_3d_array(pdb_3d)

if __name__ == '__main__':
    # Generate Hilbert Curves
    print "Generating Curves..."
    hilbert_3d = gen_hilbert_3D(6)
    hilbert_2d = gen_hilbert_2D(9)
    print "Generating Curves Done."

    # Read Ras PDBs
    pdb_files = []
    for line in sorted(os.listdir('Ras-Gene-PDB-Files')): pdb_files.append(line)
    if debug: print "Total PDB Entries:", len(pdb_files)

    # Process PDB Entries
    for pdb_file in pdb_files:
        pdb_data = get_pdb_data('Ras-Gene-PDB-Files/'+ pdb_file)
        pdb_3d_model = gen_3d_model(pdb_data, 70, -70, 2.1875)
        encoded_pdb_2d = encode_3d_pdb(pdb_3d_model, hilbert_3d, hilbert_2d)
