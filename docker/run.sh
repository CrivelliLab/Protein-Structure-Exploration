#!/usr/bin/env bash
#
#- Launches Docker
# Updated: 7/18/17

# PATHs
PROJECT="$(dirname "$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )")"

# Variables
IMG=hpc-deeplearning:latest

# Build Protein-Structure-Exploration:GPU
docker run -v $PROJECT:/home/project -ti $IMG
