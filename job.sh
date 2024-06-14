#!/bin/bash
#SBATCH --mem=40GB  # Requested Memory
#SBATCH -p gpu  # Partition
#SBATCH --constraint a100 
#SBATCH --gres=gpu:1  # Number of GPUs
#SBATCH -t 4:00:00  # Job time limit
#SBATCH --mail-type=BEGIN
#SBATCH -o slurm-%j.out  # %j = job ID

module load miniconda/22.11.1-1
conda activate SelfRewardRAGEnv
module load cuda/12.2.1

python src/evaluation.py