#!/usr/bin/env bash
# nvidia-run.sh
#- Launches Nvidia Docker
# Updated: 09/25/17

# PATHs
PROJECT="$(dirname "$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )")"

# Variables
#IMG=rzamora4:rzamora4/hpc-deeplearning
IMG=dl-docker:latest

# Build Protein-Structure-Exploration:GPU
nvidia-docker run -v $PROJECT:/home/project -ti $IMG
