#!/bin/bash
#SBATCH --job-name=salbp_batch
#SBATCH --nodes=1
#SBATCH --partition=standard
#SBATCH --gres=gpu:v100:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=14
#SBATCH --time=3:00:00
#SBATCH --output=logs/salbp_output_%j.log
#SBATCH --error=logs/salbp_error_%j.log

module load anaconda
source activate cuopt_env

python solve_salbp_scholl.py