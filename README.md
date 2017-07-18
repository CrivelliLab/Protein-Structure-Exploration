# Deep Learning-Enabled Protein Structure Exploration

Summer VFP Project

Last Updated: 7/17/17

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
useful for learning on generated datasets accross three different types of harware,
including typcial Ubuntu desktop machines, NERSC's Cori, and OLCF's DGX-1.

This project is still evolving rapidly and this documentation, as well as the
code contained in this repository, is subject to rapid change. 

## Getting Started

For more information about the **project and methodology**:

- [Jupyter Notebook](notebooks/Deep-Learning-Enabled-Protein-Structure-Exploration.ipynb) - project summary and interactive visualizations
- [Directory Structure](docs/directory_structure.md) - diagram of project files and directories
- [Project Citations](docs/project_citations.md) - work cited

For more information on **setup and installation**:

- [Local Workstation](docs/setup_local.md) - for small-scale data processing and network training
- [NERSC Edison](docs/setup_nersc.md) - for large-scale data processing
- [OLCF DGX1](docs/setup_olcf.md) - for large-scale network training

For more information about **workflows**:

- [Data Processing on Local Workstation](docs/processing_workflow_local.md)
- [Data Processing on NERSC Edison](docs/processing_workflow_nersc.md)
- [Network Training on Local Workstation](docs/training_workflow_local.md)
- [Network Training on OLCF DGX1](docs/training_workflow_olcf.md)

For more information about **experiments and results**:

- [Binary Classification of RAS and WD40](reports/experiments/binary_classification.md)
- [Binary Classification of RAS PSI-BLAST Search](reports/experiments/blast_classification.md)
- [Multi-class Classification of RAS and RAS Related Families](reports/experiments/multi_classification.md)
