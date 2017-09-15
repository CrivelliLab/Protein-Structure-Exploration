# Experiment 0001 - 3D -> 2D encoded ShapeNetVox64

Updated: 8/10/17

## Data

The ShapeNetVox64 voxel data for 13 classes of objects are converted to 512 x 512 resolution
mono-channeled images using 3D to 2D hilbert curve mappings.

Link to data: --

## Models

### > Model 0 - CIFAR_512_1CHAN
Keras neural network definition inspired by CIFAR 10 image
recognition network. Network utilizes convolutional layers to do
multi-class classification between the different images.

## Results

Average time per epoch: 723 secs
