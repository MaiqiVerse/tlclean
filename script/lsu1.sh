#!/bin/bash
#SBATCH -J explain
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH -n 16
#SBATCH --time=72:00:00
#SBATCH --partition=gpu         
#SBATCH --gres=gpu:1         
#SBATCH -o slurm-%j.out
#SBATCH -e slurm-%j.err

"$@"