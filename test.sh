#!/bin/sh
#SBATCH -p edu-20h
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=1024M
#SBATCH -N 1
#SBATCH -t 0-18:00

module load cuda

python --version

##source /home/davide.cavicchini/SIV/conda.sh
##source /home/davide.cavicchini/anaconda3/bin/activate SIV_hpe


