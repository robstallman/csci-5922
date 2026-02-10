#!/bin/bash

#SBATCH --account=ucb722_asc1
#SBATCH --partition=aa100
#SBATCH --gres=gpu:1
#SBATCH --job-name=csci5922_lab2_modeltraining
#SBATCH --output=slurm_job_logs/csci5922_lab2_modeltraining.%j.out
#SBATCH --time=12:00:00
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --mail-type=ALL
#SBATCH --mail-user=rost5691@colorado.edu

export IMAGES=/projects/rost5691/containers/
export WORKDIR=/home/rost5691/projects/csci-5922
cd $WORKDIR
apptainer exec --nv -B $WORKDIR:/csci-5922,$WORKDIR/models:/csci-5922/models,$WORKDIR/training_curves:/csci-5922/training_curves,$WORKDIR/data:/csci-5922/data,$WORKDIR/logs:/csci-5922/logs,$WORKDIR/wandb:/csci-5922/wandb $IMAGES/pytorch-2.9.1-cuda12.8.sif python3 $WORKDIR/train_deep_network.py