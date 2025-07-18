#!/bin/bash
#SBATCH --job-name=lumin_test
#SBATCH --output=lumin_test_%j.out
#SBATCH --error=lumin_test_%j.err
#SBATCH --partition=short,slurm
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --time=00:20:00
#SBATCH --mem=2G

# load necessary modules
module purge
module load intel/oneapi_base openmpi/5.0.7

# run MPI program
mpirun -np 4 main