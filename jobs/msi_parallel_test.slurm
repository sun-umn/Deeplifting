#!/bin/bash -l
#SBATCH --time=48:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --mem=20g
#SBATCH --tmp=10g
#SBATCH --mail-type=ALL
#SBATCH --mail-user=dever120@umn.edu
#SBATCH -p apollo_agate
cd /home/jusun/dever120/Deeplifting
export PATH=/home/jusun/dever120/miniconda3/envs/deeplifting/bin:$PATH
python -m tasks test-parallel