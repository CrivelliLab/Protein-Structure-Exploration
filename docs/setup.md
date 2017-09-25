# Setup and Installation
Updated: 09/25/17

The following document details how to set up a local version of repository and
environment for deep neural network training.

## Cloning Repository from Github

```
$ git clone -

```

## Building and Running Deep-Learning Docker

> NOTE: Docker or Nvidia-docker must be installed for CPU or GPU systems
correspondingly. For more information on how to install Docker:
[Installing Docker]().

- Build docker

```
$ cd docker
$ source build.sh

```

- Run docker

```
$ source run.sh # For CPU Systems
OR
$ source nvidia-run.sh # For GPU Systems

```

- Activate Conda Environment

```
$ source activate deeplearning-cpu # For CPU Systems
OR
$ source activate deeplearning-gpu # For GPU Systems

```
