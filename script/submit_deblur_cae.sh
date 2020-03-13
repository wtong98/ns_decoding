#!/bin/bash
# Tensorflow with GPU support example submit script for Slurm.
#
# Replace <ACCOUNT> with your account name before submitting.
#
#SBATCH -A stats           # Set Account name
#SBATCH --job-name=cae_training  # The job name
#SBATCH -c 1                   # Number of cores
#SBATCH -t 0-6:00              # Runtime in D-HH:MM
#SBATCH --gres=gpu:1           # Request a gpu module
 
module load singularity
singularity exec --nv /moto/opt/singularity/tensorflow-1.13-gpu-py3-moto.simg python deblur_cae.py
