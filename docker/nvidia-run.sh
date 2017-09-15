#!/usr/bin/env bash
# nvidia-run.sh
#- Launches Nvidia Docker
# Updated: 8/10/17

# PATHs
PROJECT="$(dirname "$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )")"

# Variables
IMG=dl-experiments:latest

# Build Protein-Structure-Exploration:GPU
nvidia-docker run -v $PROJECT:/home/project -ti $IMG
