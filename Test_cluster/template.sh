#!/bin/bash

# Set max time to your time estimation for your program be as precise as possible
# for optimal cluster utilization
#SBATCH --time=00:05:00

#
# There are a lot of ways to specify hardware ressources we recommend this one
#

# Set ntasks to 1 except when you are trying to run 2 tasks that are exactly the same
# although in some cases like simulations with random events this may be desirable
#SBATCH --ntasks=1  # number of processor cores (i.e. tasks)

# Now set your cpu and memory requirements and onve again be as precise as possible
# keep in mind if your pregram exceeds the requested memory it will be terminated prematurely
# the configuration below will result in a job allocation of 2 cores and 40 MB of RAM
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=2000M

# If you so choose Slurm will notify you when certain events happen for your job 
# for a full list of possible options look furhter down
#SBATCH --mail-type END

# Feel free to give your job a good name to better identify it later
# the same name expansions as for the ourput and error path apply here as well
# see below for additional information
#SBATCH --job-name="<my_job_name>"

# Always try to use absolute paths for your output and error files
# IF you only specify an output file all error messages will automaticly be redirected in there
# You can utilize name expansion to make sure each job has a uniq output file if the file already exists 
# Slurm will delete all the content that was there before before writing to this file so beware.
#SBATCH --output=<slurm_dir>/output/<my_job_name>-%j.out
#SBATCH --error=<slurm_dir>/errors/<my_job_name>-%j.err

#
# Prepare your environment
#
testssh
# causes jobs to fail if one command fails - makes failed jobs easier to find with tools like sacct
set -e

# load modules
module load python/3.10

# Set variables you need
project="<project_dir>"
scratch="/scratch/$USER/$SLURM_JOB_ID"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

#Create your scratch space
mkdir -p $scratch
cd $scratch

# Copy your program (and maybe input files if you need them)
cp $project/ .

python3 <code_name> <args>

# copy results to an accessable location
# only copy things you really need
cp -r results $project/results/SLURM_JOB_ID 

# Clean up after yourself
cd
rm -rf $scratch

# exit gracefully
exit 0