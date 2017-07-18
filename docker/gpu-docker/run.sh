#!/usr/bin/env bash

# PATHs
SRC =
DATA

# Build Protein-Structure-Exploration:GPU
nvidia-docker run -v $SRC:/home/prot-struct-exp-gpu/src -v -ti protein-structure-exploration:gpu
