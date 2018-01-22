# Deep Learning-Enabled Protein Structure Exploration

Summer VFP 2017 Project

Last Updated: 09/25/17

Lead Maintainers:

- [Rafael Zamora](https://github.com/rz4) - rz4@hood.edu

- [Thomas Corcoran](https://tjosc.github.io/) - tomcorcoran@protonmail.com

This is a collection of utilities to aid in the computational exploration of
protein structures. The programs contained here represent a data pipeline that
is intended to allow a user to easily collect and transform their own dataset of
3-dimensional protein model files (i.e. PDB files) into 2-dimensional
image representations using a series of Hilbert curve mappings. Facilities are
also provided for the rendering and visualization of protein representations at
all stages of their journey throughout the pipeline. Finally, an interactive
command-line module is provided to allow for easy segmentation and
serialization of generated image datasets for eventual ingestion by
convolutional neural networks for classification and feature extraction tasks.

Beyond the data processing components, this repository also contains code
allowing for easy deployment of a variety of neural network architectures
useful for learning on generated datasets across three different types of hardware,
including typical Ubuntu desktop machines, NERSC's Cori, and OLCF's DGX-1.

This project is still evolving rapidly and this documentation, as well as the
code contained in this repository, is subject to rapid change.

## Getting Started

For more information about the **project and methodology**:

- [Jupyter Notebook](notebooks/Deep-Learning-Enabled-Protein-Structure-Exploration.ipynb) - interactive project methodology
