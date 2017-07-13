#!/bin/bash
#SBATCH --job-name="protein_encoding"
#SBATCH --output="protein_encoding.out"
#SBATCH --partition=debug
#SBATCH --nodes=2
#SBATCH -t 00:02:00

# Load Modules
module load python/2.7.5
module load VTK/5.10.1
module load mpi4py/1.3.1

# Variables
PFILE = P01111P01112P01116pos_t45.npy
C3FILE = hilbert_3d6.npy
C2FILE = hilbert_2d9.npy

# Run Job
srun -n 48 python features/EncodePDBs.py -pf $PFILE -c3 $C3FILE -c2 $C2FILE
