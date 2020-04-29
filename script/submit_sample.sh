#!/bin/sh
#
# Submit job to decode images
#
#SBATCH --account=stats # The account name for the job.
#SBATCH --job-name=decode # The job name.
#SBATCH -c 8 # The number of cpu cores to use.
#SBATCH --time=0:30:00 # The time the job will take to run.
#SBATCH --mem-per-cpu=4gb # The memory the job will use per cpu core.
 
module load anaconda
 
#Command to execute Python program
python sample.py
 
#End of script
