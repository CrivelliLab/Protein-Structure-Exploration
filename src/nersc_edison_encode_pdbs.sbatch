#!/bin/bash
#SBATCH --job-name="protein_encoding"
#SBATCH --output="protein_encoding.out"
#SBATCH --partition=debug
#SBATCH --nodes=2
#SBATCH -t 00:10:00

# Load Modules
module load python/2.7.5
module load VTK/5.10.1
module load mpi4py/1.3.1

# Variables
CORES = 48                              # Nodes * 24
PFILE = P01111P01112P01116pos_t45.npy   # Processed PDB data file
C3FILE = hilbert_3d_6.npy               # 3D SFC file
C2FILE = hilbert_2d_9.npy               # 2D SFC file

# Run Job
srun -n $CORES python src/features/EncodePDBs.py $PFILE $C3FILE $C2FILE
