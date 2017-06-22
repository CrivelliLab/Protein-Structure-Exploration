#!/bin/bash
#SBATCH --job-name="protein_encoding"
#SBATCH --output="protein_encoding.%j.%N.out"
#SBATCH --partition=regular
#SBATCH --nodes=1000
#SBATCH -t 00:08:00
#SBATCH -L project

# Run Job

module load VTK/5.10.1
module load python/2.7.5
module load mpi4py/1.3.1

cd Protein-Structure-Prediction/src
srun -n 24000 python EncodePDBs.py
