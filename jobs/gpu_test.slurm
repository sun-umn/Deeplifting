#!/bin/bash -l
#SBATCH --time=24:00:00
#SBATCH --ntasks=8
#SBATCH --mem=16g
#SBATCH --tmp=10g
#SBATCH --mail-type=ALL
#SBATCH --mail-user=dever120@umn.edu
#SBATCH -p apollo_agate
#SBATCH --gres=gpu:a100:1
cd /home/jusun/dever120/Deeplifting
export PATH=/home/jusun/dever120/miniconda3/envs/deeplifting/bin:$PATH
nvidia-smi && python -c "import torch; print(torch.cuda.device_count())" && python -m torch.utils.collect_env