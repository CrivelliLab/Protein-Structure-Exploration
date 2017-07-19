#!/usr/bin/env bash
# GPU Protein-Structure-Exploration run.sh
#- Launches Docker and Mounts Project src and data
# Updated: 7/17/17

# Build Protein-Structure-Exploration:GPU
sudo docker build -t protein-structure-exploration:gpu .
