# Directory Structures

Updated: 07/11/17

```
|--- LICENSE
|--- README.md
|--- requirements.txt
|--- data
|    |--- interim
|    |--- processed
|    |    |---PDB
|    |
|    |--- raw
|         |--- PDB
|         |--- SFC
|         |--- BLAST
|
|--- docs
|    |--- citations.md
|    |--- directory_structure.md
|    |--- workflow.md
|
|--- models
|
|--- notebooks
|    |--- Deep-Learning-Enabled-Protein-Structure-Exploration.ipynb
|
|--- reports
|    |--- figures
|    |--- experiments
|
|--- src
     |--- __init__.py
     |--- run_encode_pdbs.sbatch
     |--- data
     |    |--- BlastSearch.py
     |    |--- GenSFCs.py
     |    |--- GenPDBs.py
     |
     |--- features
     |    |--- EncodePDBs.py
     |    |--- ProcessPDBs.py
     |
     |--- models
     |    |--- CIFAR_512.py
     |    |--- ModelTrainer.py
     |    |--- SplitProcessed.py
     |
     |--- analysis
     |    |--- GenSaliency.py
     |
     |--- validation
     |    |--- ValRotations.py
     |    |--- ValSFCs.py
     |
     |--- visualization
          |--- Visualizations.py

```
