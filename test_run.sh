#!/bin/bash
#SBATCH --job-name=training
#SBATCH --partition=main
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

python -m src.main
