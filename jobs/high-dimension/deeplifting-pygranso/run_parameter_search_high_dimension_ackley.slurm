#!/bin/bash -l
#SBATCH --time=96:00:00
#SBATCH --ntasks=8
#SBATCH --mem=40g
#SBATCH --tmp=10g
#SBATCH --mail-type=ALL
#SBATCH --mail-user=dever120@umn.edu
#SBATCH -p apollo_agate
cd /home/jusun/dever120/Deeplifting
export PATH=/home/jusun/dever120/miniconda3/envs/deeplifting-cpu/bin:$PATH
python -m tasks find-best-deeplifting-architecture --problem_series='ackley' --method='pygranso' --dimensionality='high-dimensional'