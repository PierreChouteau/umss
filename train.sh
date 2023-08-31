#!/bin/bash    

#SBATCH --job-name=train
#SBATCH --partition=V100
#SBATCH --cpus-per-task=10
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --nodelist=node10
#SBATCH --time=48:00:00
#SBATCH --output=./train_files/train%j.out
#SBATCH --error=./train_files/train%j.err

# Load conda environment
MODULE_ENV="umss"
eval "$(conda shell.bash hook)"
conda activate $MODULE_ENV

# activer l'écho des commandes
set -x

# run script
python train.py -c config.txt
