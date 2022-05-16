#!/bin/bash
#SBATCH -N1
#SBATCH -c8
#SBATCH --mem=4gb
#SBATCH --time=168:00:00

#SBATCH --job-name=QML

python3 QML_DQN_FROZEN_LAKE.py
