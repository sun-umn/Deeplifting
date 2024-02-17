#!/bin/bash -l
#SBATCH --time=48:00:00
#SBATCH --ntasks-per-node=8
#SBATCH --mem=65g
#SBATCH --tmp=20g
#SBATCH --mail-type=ALL
#SBATCH --mail-user=dever120@umn.edu
#SBATCH -p apollo_agate
cd /home/jusun/dever120/Deeplifting
export PATH=/home/jusun/dever120/miniconda3/envs/deeplifting/bin:$PATH
python -m tasks test-parallel