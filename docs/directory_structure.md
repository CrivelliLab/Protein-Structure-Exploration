# Directory Structures

Updated: 07/11/17

```
|--- LICENSE
|--- README.md
|--- requirements.txt
|--- data
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
|         |--- UNIPROT
|         |--- BLAST
|
|--- docs
|    |--- CITATIONS
|
|--- models
|    |--- CLASSER_RAS_WD40_MD512_HH_3CHAN.hdf5
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
