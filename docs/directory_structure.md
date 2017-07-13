# Directory Structure

Updated: 07/12/17

```
|--- README.md ~ Setup and installation
|--- data
|    |--- interim ~ Contains processed PDB .npy files
|    |--- processed
|    |    |---tars ~ Contains encoded PDB .tar.gz files
|    |    |---datasets ~ Contains segmented encoded PDB training datasets
|    |
|    |--- raw
|         |--- PDB ~ Contains PDB ids .txt files and folders containing .pdb.gz files
|         |--- SFC ~ Contains space filling curve .npy files
|         |--- BLAST ~ Contains BLAST search result .csv files of Protein Data Bank
|
|--- docs
|    |--- project_citations.md
|    |--- directory_structure.md 
|    |--- processing_workflow_local.md
|    |--- processing_workflow_nersc.md
|    |--- setup_local.md
|    |--- setup_nersc.md
|    |--- setup_olcf.md
|    |--- training_workflow_local.md
|    |--- training_workflow_olcf.md
|
|--- models ~ Contains model folders with network Architecture, network weights and training results
|
|--- notebooks
|    |--- Deep-Learning-Enabled-Protein-Structure-Exploration.ipynb ~ Project summary and methodology
|
|--- reports
|    |--- figures ~ Project images and graphs
|    |--- deliverables ~ Project paper and poster
|    |--- experiments ~ documentation for experiments
|
|--- src
     |--- __init__.py
     |--- run_encode_pdbs.sbatch ~ SLURM batch file for running large parallel PDB encoding jobs
     |--- data
     |    |--- ParseBLAST.py ~ Parses BLAST search results .csv and creates PDB id list .txt file
     |    |--- GenSFCs.py ~ Generates space filling curve .npy file
     |    |--- GetPDBs.py ~ Fetches pdb.gz files from PDB ids listed in .txt file
     |
     |--- features
     |    |--- EncodePDBs.py ~ Encodes processed PDB .npy files
     |    |--- ProcessPDBs.py ~ Processes pdb.gz files and creates processed .npy file
     |
     |--- models
     |    |--- CIFAR_512.py ~ Keras network definition of a CIFAR10 CNN able to process 512x512 images
     |    |--- ModelTrainer.py ~ Trains networks
     |    |--- SplitProcessed.py ~ Segments encoded PDB tar.gz files for network training
     |
     |--- analysis
     |    |--- GenSaliency.py ~ Generates attention maps from trained networks
     |
     |--- validation
     |    |--- ValRotations.py ~ Assesses similarity between encodings for rotation augmentation strategy
     |    |--- ValSFCs.py ~ Assess how 3D to 2D mapping affects relative distance between points
     |
     |--- visualization
          |--- Visualizations.py ~ Used to visualize protein models and encodings

```
