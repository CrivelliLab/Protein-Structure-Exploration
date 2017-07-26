#!/usr/bin/env bash
#
#- Launches Docker
# Updated: 7/18/17

# PATHs
PROJECT="$(dirname "$(dirname "$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )")")"

# Variables
IMG=hpc-deeplearning:latest

# Build Protein-Structure-Exploration:GPU
docker run -ti $IMG
