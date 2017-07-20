#!/usr/bin/env bash
# CPU Protein-Structure-Exploration run.sh
#- Launches Docker and Mounts Project src and data
# Updated: 7/18/17

# PATHs
# PATHs
PROJECT="$(dirname "$(dirname "$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )")")"
SRC=$PROJECT"/src"
DATA=$PROJECT"/data"
MODELS=$PROJECT"/models"
DSRC=/home/protein-structure-exploration/src
DDATA=/home/protein-structure-exploration/data
DMODELS=/home/protein-structure-exploration/models

# Variables
IMG=protein-structure-exploration:cpu

# Build Protein-Structure-Exploration:GPU
docker run -v $SRC:$DSRC -v $DATA:$DDATA -v $MODELS:$DMODELS -ti $IMG
