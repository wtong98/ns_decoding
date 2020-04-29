#!/bin/bash
# Tensorflow with GPU support example submit script for Slurm.
#
# Replace <ACCOUNT> with your account name before submitting.
#
#SBATCH -A stats           # Set Account name
#SBATCH --job-name=cae_training  # The job name
#SBATCH --exclusive
#SBATCH --nodes=1
#SBATCH --mem=32gb
#SBATCH -t 3-00:00              # Runtime in D-HH:MM
 
module load singularity
singularity exec --nv /moto/opt/singularity/tensorflow-1.13-gpu-py3-moto.simg python deblur_cae.py
