#!/bin/bash -l
#SBATCH --time=48:00:00
#SBATCH --ntasks=1
#SBATCH --mem=40g
#SBATCH --tmp=10g
#SBATCH --mail-type=ALL
#SBATCH --mail-user=dever120@umn.edu
#SBATCH -p apollo_agate
#SBATCH --gres=gpu:a100:1
cd /home/jusun/dever120/Deeplifting
export PATH=/home/jusun/dever120/miniconda3/envs/deeplifting/bin:$PATH
python -m deeplifting.svm run-svm-deeplifting --trials=10 --experimentation=True