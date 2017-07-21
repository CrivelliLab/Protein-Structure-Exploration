#!/bin/bash -l
#SBATCH -C haswell
#SBATCH --job-name="protein_encoding"
#SBATCH --output="protein_encode.out"
#SBATCH --partition=regular
#SBATCH --nodes=5
#SBATCH -t 00:45:00

# Load Modules
module load python/2.7-anaconda
source activate myenv

# Variables
CORES=240                              # Nodes * 48
PFILE=RASCLEANBOUNDED0%64_t45.npy   # Processed PDB data file
C3FILE=hilbert_3d_6.npy                # 3D SFC file
C2FILE=hilbert_2d_9.npy                # 2D SFC file

# Run Job
mpirun -n $CORES python src/features/EncodePDBs.py -sb [-34,34] $PFILE $C3FILE $C2FILE
