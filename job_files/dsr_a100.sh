#!/bin/bash

#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --job-name=dsr_a100
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --time=05:00:00
#SBATCH --output=logs/dsr_a100_%j.out

module purge
module load 2024
module load Anaconda3/2024.06-1

source activate dso

cd $HOME/nesymres/benchmark_others

python bench_dsr.py
