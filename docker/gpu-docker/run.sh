#!/usr/bin/env bash
# GPU Protein-Structure-Exploration run.sh
#- Launches Docker and Mounts Project src and data
# Updated: 7/17/17

# PATHs
PROJECT="$(dirname "$(dirname "$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )")")"
SRC=$PROJECT"/src"
DATA=$PROJECT"/data"
MODELS=$PROJECT"/models"
DSRC=/home/protein-structure-exploration-gpu/src
DDATA=/home/protein-structure-exploration-gpu/data
DMODELS=/home/protein-structure-exploration-gpu/models

# Variables
IMG=rzamora4/protein-structure-exploration:gpu

# Build Protein-Structure-Exploration:GPU
nvidia-docker run -v $SRC:$DSRC:Z -v $DATA:$DDATA:Z -v $MODELS:$DMODELS:Z -ti $IMG
