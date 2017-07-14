# Local Workstation Data Processing

Updated: 7/13/17

## Introduction

This document details the workflow of processing Protein Data Bank (PDB) files
into 2-D representations on a local machine. The workflow takes in a .txt file
of PDB ids and outputs a folder containing 2D encoded PDB images.

This workflow can be used on small data sets. For processing large amounts of
data refer to
...

## Procedures

### Generating 2D Encodings of PDBs:

1. Create a line seperated .txt file of PDB ids for all PDBs you would like to
process. Save file in the [data/raw/PDB](../data/raw/PDB) directory and with the
following naming convention: **CLASS_ids.txt**

2. Fetch PDB files from the database using the [src/data/GetPDBs.py](../src/data/GetPDBs.py)
script. This will create a folder in [data/raw/PDB](../data/raw/PDB) named
according to the filename of the PDB ids .txt file.

```
# Fetch PDBs defined in CLASS_ids.txt
#

:Prot-Struct-Explor/$ python src/data/GetPDBs.py CLASS_ids.txt
Reading PDB List...
CLASS contains 100 entries...
Fetching PDBs...
100%|████████████████████████████████████████| 100/100 [00:00<00:00, 457.39it/s]
PDBs saved in: data/raw/PDB/CLASS

```

3. Generate space filling curves which will be used for the 3D to 2D mapping.
This can be done by running the [src/data/GenSFCs.py](../src/data/GenSFCs.py)
script which will allow you to define the type and order of the curve that is
generated. The generated curves will be saved in [data/raw/SFC](../data/raw/SFC).
You will need to generate both a 3D and 2D space filling curve. The following
curves are available:

  - hilbert_3d
  - hilbert_2d
  - z_curve_3d
  - z_curve_2d

> ***Important:*** Select the order of curves such that the 3D curve is mapped
onto the 2D curve in a one to one manner. The following table lists possible
mappings:

| 3D Order | 2D Order | 3D Curve Shape | 2D Curve Shape |
|:--------:|:--------:|:--------------:|:--------------:|
| 2        | 3        | 4 x 4 x 4      | 8 x 8
| 4        | 6        | 16 x 16 x 16   | 64 x 64
| 6        | 9        | 64 x 64 x 64   | 512 x 512
| 8        | 12       | 256 x 256 x256 | 4096 x 4096

```
# Generate 3d hilbert curve of order 6
# Generate 2d hilbert curve of order 9

::Prot-Struct-Explor/$ python src/data/GenSFCs.py hilbert_3d 6
Generating Curve...
Curve Saved In: data/raw/SFC/hilbert_3d_6.npy

::Prot-Struct-Explor/$ python src/data/GenSFCs.py hilbert_2d 9
Generating Curve...
Curve Saved In: data/raw/SFC/hilbert_2d_9.npy

```

4. Parse the atomic coordinate information for desired channels from the PDBs.
This can be done by running the [src/features/ProcessPDBs.py](../src/features/ProcessPDBs.py)
script which allows you to define the channels as well as the rotation augmentation
strategy. The processed PDB data will be save in [data/interim/](../data/interim/)
with the following naming convention: **CLASS_tN.npy** , where n is the rotation
augmentation.

```
# Process hydrophobic, polar, and charged channels of CLASS PDBs
# with 45 degree rotation augmentations

:Prot-Struct-Explor/$ python src/features/ProcessPDBs.py CLASS 45 'hydrophobic,polar,charged'
Read PDB Ids in: data/raw/PDB/CLASS/
Processed data saved in: data/interim/CLASS_t45.npy

```

5. Encode the processed PDB data into 2D by running the
[src/features/encodePDBs.py](../src/features/encodePDBs.py) script. The encoded 2D
data will save in [data/processed/tars/](../data/processed/tars) in a folder
named after the encoding parameters.

```
# Encode processed CLASS PDB data using an order 6 3D hilbert curve
# and an order 9 2D hilbert curve

:src/$ python features/EncodePDBs.py CLASS_t45.npy hilbert_3d_6.npy hilbert_2d_9.npy
Processing: data/interim/CLASS_t45.npy
MPI Cores: 1
Encodings saved in: data/processed/tars/CLASS_t45_MD_HH512

```

>**Note**: [src/features/encodedPDBs.py](../src/features/encodePDBs.py) is implemented
to run in parallel through MPI.

```
# Encode processed CLASS PDB data using MPI on 4 cores
#

:Prot-Struct-Explor/$ mpirun -n 4 python srcfeatures/EncodePDBs.py CLASS_t45.npy hilbert_3d_6.npy hilbert_2d_9.npy
Processing: data/interim/CLASS_t45.npy
MPI Cores: 4
Encodings saved in: data/processed/tars/CLASS_t45_MD_HH512

```

6. [Optional] Compress encodings folder using tar. Change directory to
[data/processed/tars/](../data/processed/tars) and run ```tar -zcf```.

```
# Tar Encodings
#

:Prot-Struct-Explor/$ cd data/processed/tars
:Prot-Struct-Explor/data/processed/tars/$ tar -zcf CLASS_t45_MD_HH512 CLASS_t45_MD_HH512.tar.gz

```
