# NERSC Cori PDB Data Processing

Updated: 7/14/17
[NOT PASSING] - Out of date
                |- Running with Docker/Shifter

## Introduction

This document details the workflow of processing Protein Data Bank (PDB) files
into 2-D representations on the NERSC Edison system. The workflow takes in a .txt file
of PDB ids and outputs a folder containing 2D encoded PDB images.

This workflow is for processing large amounts of data. For processing small amounts of
data refer to the workflow on a [local workstation](processing_workflow_local.md).

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
CLASS contains 300 entries...
Fetching PDBs...
100%|████████████████████████████████████████| 300/300 [00:00<00:00, 457.39it/s]
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

5. [Optional] Run a profiling encoding job on a debug node using the
[src/nersc_edison_profile_encode.sbatch](../src/nersc_edison_profile_encode.sbatch)
SLURM script. This will provide a rough estimation of the run time of the needed
to process all encodings on **n** number of nodes. Using a text editor, change
the parameters within the SBATCH file.

> ***Important:*** By default, the script will encode the data using the atomic
space filling model of the data. To encode only the skeleton of the model add
the ```-sk``` flag. Also, the script by default bounds the protein models dynamically.
To use static bounds define with comma seperated values using the ```-sb``` flag
(ex. ```-sb '-100,100'```).

***nersc_edison_profile_encode.sbatch***
```bash
#!/bin/bash
#SBATCH --job-name="profile_protein_encoding"
#SBATCH --output="profile_encode.out"
#SBATCH --partition=debug
#SBATCH --nodes=10
#SBATCH -t 00:05:00

# Load Modules
module load python/2.7.5
module load VTK/5.10.1
module load mpi4py/1.3.1

# Variables
CORES = 240                              # Nodes * 24
PFILE = CLASS_t45.npy                    # Processed PDB data file
C3FILE = hilbert_3d_6.npy                # 3D SFC file
C2FILE = hilbert_2d_9.npy                # 2D SFC file

# Run Job
srun -n $CORES python src/features/EncodePDBs.py -p $PFILE $C3FILE $C2FILE

```

After changing variables, submit SLURM Job. Results of profiling will
output to profile_encode.out .

```
# Submit SLURM job and display results
#

:Prot-Struct-Explor/$ sbatch src/nersc_edison_profile_encode.sbatch
Job Summitted.
:Prot-Struct-Explor/$ cat profile_encode.out

```

>***CAUTION:*** This is a very rough estimation from a sample size of the number of
available cores. The average encoding time approaches the true average with more cores.
Be sure to allocate more time than the estimation to be safe.

6. Encode the processed PDB data into 2D by running the
[src/nersc_edison_encode_pdbs.sbatch](../src/nersc_edison_encode_pdbs.sbatch) script.
The encoded 2D data will save in [data/processed/tars/](../data/processed/tars)
in a folder named after the encoding parameters. Using a text editor, change the
parameters within the SBATCH file including the allocated time.

***nersc_edison_encode_pdbs.sbatch***
```bash
#!/bin/bash
#SBATCH --job-name="protein_encoding"
#SBATCH --output="protein_encoding.out"
#SBATCH --partition=regular
#SBATCH --nodes=10
#SBATCH -t 00:45:00                     # Be sure to change to estimated time

# Load Modules
module load python/2.7.5
module load VTK/5.10.1
module load mpi4py/1.3.1

# Variables
CORES = 240                             # Nodes * 24
PFILE = CLASS_t45.npy                   # Processed PDB data file
C3FILE = hilbert_3d_6.npy               # 3D SFC file
C2FILE = hilbert_2d_9.npy               # 2D SFC file

# Run Job
srun -n 48 python src/features/EncodePDBs.py $PFILE $C3FILE $C2FILE

```

After changing variables, submit SLURM Job.

```
# Submit SLURM job
#

:Prot-Struct-Explor/$ sbatch src/nersc_edison_profile_encode.sbatch
Job Summitted.

```

7. [Optional] Compress encodings folder using tar. Change directory to
[data/processed/tars/](../data/processed/tars) and run ```tar -zcf```.

>**Note:** Compressing is only needed to move around the encoded data. If network
training will be done on same system, this step is not necessary.

```
# Tar encodings
#

:Prot-Struct-Explor/$ cd data/processed/tars
:Prot-Struct-Explor/data/processed/tars/$ tar -zcf CLASS_t45_MD_HH512 CLASS_t45_MD_HH512.tar.gz

```
