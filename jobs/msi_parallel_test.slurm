#!/bin/bash -l
#SBATCH --time=72:00:00
#SBATCH --ntasks-per-node=8
#SBATCH --mem=120g
#SBATCH --tmp=10g
#SBATCH --mail-type=ALL
#SBATCH --mail-user=dever120@umn.edu
#SBATCH -p apollo_agate
cd /home/jusun/dever120/Deeplifting
export PATH=/home/jusun/dever120/miniconda3/envs/deeplifting/bin:$PATH
export OMP_NUM_THREADS=1
python -m tasks test-parallel