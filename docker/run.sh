#!/usr/bin/env bash
# run.sh
#- Launches Docker
# Updated: 8/10/17

# PATHs
PROJECT="$(dirname "$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )")"

# Variables
IMG=dl-experiments:latest

# Build Protein-Structure-Exploration:GPU
docker run -v $PROJECT:/home/project -ti $IMG
