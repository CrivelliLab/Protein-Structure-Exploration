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

# Run Job
mpirun -n $CORES python encode_pdb.py
