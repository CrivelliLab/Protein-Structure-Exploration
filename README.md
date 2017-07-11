# Deep-Learning-Enabled-Protein-Structure-Exploration
A collection of utilities to aid in the computational evaluation of native
computationally-generated protein structures.

## Directory Structures

```
|--- LICENSE
|--- README.md
|--- requirements.txt
|--- data
|    |--- external
|    |--- interim
|    |    |--- RAS_t45.npy
|    |    |--- WD40_t45.npy
|    |    
|    |--- processed
|    |    |---PDB
|    |
|    |--- raw
|         |--- PDB
|         |    |--- ALL
|         |    |    |--- all_pdb_ids_minus_ras.txt
|         |    |    |--- all_pdb_ids_minus_wd40.txt
|         |    |    |--- all_pdb_ids_no_dupes.txt
|         |    |    
|         |    |--- RAS
|         |    |    |--- ras_pdb_ids.txt
|         |    |--- WD40
|         |         |--- wd40_pdb_ids.txt
|         |
|         |--- SFC
|         |--- BLAST
|
|--- docs
|--- models
|    |--- CLASSER_RAS_WD40_MD512_HH_3CHAN.hdf5
|
|--- notebooks
|    |--- Deep-Learning-Enabled-Protein-Structure-Exploration.ipynb
|
|--- references
|    |--- CITATIONS
|
|--- reports
|    |--- figures
|
|--- src
     |--- __init__.py
     |--- data
     |    |--- BlastSearch.py
     |    |--- PfamSearch.py
     |    |--- GenSFCs.py
     |    |--- GenPDBs.py
     |
     |--- features
     |    |--- EncodePDBs.py
     |    |--- ProcessPDBs.py
     |    |--- run_encode_pdbs.sbatch
     |
     |--- models
     |    |--- PDBClassifier.py
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
