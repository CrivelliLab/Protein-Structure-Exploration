# Local Workstation Data Processing

Updated: 7/12/17

## Introduction

This document details the workflow of processing Protein Data Bank (PDB) files
into 2-D representations on a local machine.

...

## Procedures

### Generating 2D Encodings of PDBs:

1. Create a line seperated .txt file of PDB ids for all PDBs you would like to
process. Save file in the [data/raw/PDB]() directory and with the following
naming convention: **LABEL_ids.txt**

2. Change directory to [src/]() and fetch PDB files from the database using
the [data/GetPDBs.py]() script. This will create a folder in [data/raw/PDB]()
named **LABEL** according to the filename of the PDB ids .txt file.

```
:Protein-Structure-Prediction/$ cd src
:src/$ python data/GetPDBs.py LABEL_ids.txt
Reading PDB List...
LABEL contains 100 entries...
Fetching PDBs...
100%|████████████████████████████████████████| 100/100 [00:00<00:00, 457.39it/s]
PDBs saved in: data/raw/PDB/LABEL

```

3. Generate space filling curves which will be used for the 3D to 2D mapping.
This can be done by running the [data/GenSFCs.py]() script which will allow you to
define the type and order of the curve that is generated. The generated curves
will be saved in [data/raw/SFC](). You will need to generate both a 3D and 2D
space filling curve. The following curves are available:

  - hilbert_3d
  - hilbert_2d
  - z_curve_3d
  - z_curve_2d

> ***Important:*** Select the order of curves such that the 3D curve is mapped onto the
2D curve in a one to one manner. The following table lists possible mappings:

| 3D Order | 2D Order | 3D Curve Shape | 2D Curve Shape |
|:--------:|:--------:|:--------------:|:--------------:|
| 2        | 3        | 4 x 4 x 4      | 8 x 8
| 4        | 6        | 16 x 16 x 16   | 64 x 64
| 6        | 9        | 64 x 64 x 64   | 512 x 512
| 8        | 12       | 256 x 256 x256 | 4096 x 4096

```
:src/$ python data/GenSFCs.py hilbert_3d 6
Generating Curve...
Curve Saved In: data/raw/SFC/hilbert_3d_6.npy

:src/$ python data/GenSFCs.py hilbert_2d 9
Generating Curve...
Curve Saved In: data/raw/SFC/hilbert_2d_9.npy

```

4. Parse the atomic coordinate information for some desired channels from the PDBs.
This can be done by running the [features/ProcessPDBs.py]() script which allows you to
define the channels as well as the rotation augmentation strategy. The processed PDB
data will be save in [data/interim/]() with the following naming convention:
**LABEL_tN.npy** where n is the rotation augmentation

```

```
