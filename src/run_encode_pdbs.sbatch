#!/bin/bash
#SBATCH --job-name="protein_encoding"
#SBATCH --output="protein_encoding.out"
#SBATCH --partition=debug
#SBATCH --nodes=2
#SBATCH -t 00:02:00
#SBATCH -L project

# Run Job
module load VTK/5.10.1
module load python/2.7.5
module load mpi4py/1.3.1

srun -n 48 python features/EncodePDBs.py -pf P01111P01112P01116pos_t45.npy -c3 hilbert_3d6.npy -c2 hilbert_2d9.npy
