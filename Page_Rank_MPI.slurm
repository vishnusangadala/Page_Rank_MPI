#!/bin/bash
#SBATCH --job-name=pagerank
#SBATCH --partition=short
#SBATCH --ntasks=5
#SBATCH --time=00:30:00
#SBATCH --output=/gpfs/projects/AMS598/projects2025_submission/project2_submission/Sangadala_Vishnu/pagerank_%j.out
#SBATCH --error=/gpfs/projects/AMS598/projects2025_submission/project2_submission/Sangadala_Vishnu/pagerank_%j.err

module load python/3.10
module load mpi4py

cd /gpfs/projects/AMS598/projects2025_submission/project2_submission/Sangadala_Vishnu/

srun python3 pagerank_mpi.py
