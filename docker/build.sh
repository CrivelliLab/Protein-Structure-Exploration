#!/usr/bin/env bash
# CPU Protein-Structure-Exploration run.sh
#- Launches Docker and Mounts Project src and data
# Updated: 7/26/17

# Build Protein-Structure-Exploration:GPU
sudo docker build -t hpc-deeplearning .
sudo docker tag hpc-deeplearning:latest rzamora4/hpc-deeplearning
sudo docker ps -aq --no-trunc | xargs docker rm
sudo docker images -q --filter dangling=true | xargs docker rmi
