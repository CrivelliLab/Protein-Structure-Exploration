'''
ParseBlast.py
Updated: 07/20/17
[PASSING]

README:

The following script is used to parse Psi-Blast search results and return a list
of unique ids for protein structures in the Protein Data Bank.

Global variables used to parse the Blast results are defined under #- Global variables.
'blast_results' defines the filename of the BLAST search results. File must be in
.csv format and located in data/raw/BLAST.

'positive_uniprots' defines the list ofuniprot sequences used to divide blast
results into positive and negative sets. If result PDB is within the same Pfam
as the UniProt set, it will be considered a positive case.

Command Line Interface:

$ python ParseBLAST.py [-h] blast_results positive_uniprots

Note: Pfam search will sometimes return with an error or unknown, these cases
will be stored in an unknown set.

PDB id list .txt files will be save under data/raw/PDB with the following
naming conventions:

- <positive_uniprots>pos.csv - positive set
- <positive_uniprots>neg.csv - negative set
- <positive_uniprots>unk.csv - unkown set

'''
import os, shutil, argparse
from tqdm import tqdm

# PDB Parsing
from prody import searchPfam, pathPDBFolder, parsePDBHeader, confProDy
confProDy(verbosity='none')

#- Global Variables
blast_results = ''
positive_uniprots = []

#- Verbose Settings
debug = True
blast_results_usage = "BLAST hit table results .csv; .csv file must be in data/raw/BLAST/"
positive_uniprots_usage = "UniProt IDs for Positive Cases; comma seperated values"

################################################################################

if __name__ == '__main__':

    # Cmd Line Args
    parser = argparse.ArgumentParser()
    parser.add_argument('blast_results', help=blast_results_usage, type=str)
    parser.add_argument('positive_uniprots', help=positive_uniprots_usage, type=str)
    args = vars(parser.parse_args())
    blast_results = args['blast_results']
    positive_uniprots = args['positive_uniprots'].split(',')

    # File Paths
    os.chdir(os.path.dirname(os.path.realpath(__file__)))
    blast_results = '../../data/raw/BLAST/' + blast_results
    pdb_folder = '../../data/raw/PDB/'
    pdb_id_file = ''.join(positive_uniprots)
    if not os.path.exists(pdb_folder+'temp'): os.makedirs(pdb_folder+'temp')
    pathPDBFolder(pdb_folder+'temp')

    # Read Blast Results
    if debug: print("Parsing Blast Results...")
    hits = []
    with open(blast_results) as f:
        for line in f.readlines():
            cols = line.split(',')
            if len(cols) > 2:
                cols = cols[1].split('|')
                pdb_id = cols[3].lower() + cols[4][0]
                hits.append(pdb_id)
    hits = list(set(hits)) # remove duplicates

    # Get Set Of Positive Families
    if debug:
        print("Fetching Postive Pfam Ids...")
        pbar = tqdm(total=len(positive_uniprots))
    pos_pfam = []
    for i in range(len(positive_uniprots)):
        pfam = searchPfam(positive_uniprots[i]).keys()[0]
        pos_pfam.append(pfam)
        if debug: pbar.update(1)
    pos_pfam = list(set(pos_pfam)) # remove duplicates
    if debug: pbar.close()

    # Get Pfam Id For Hits
    if debug:
        print("Fetching Hits Pfam Ids...")
        pbar = tqdm(total=len(hits))
    hits_pfam = []
    for i in range(len(hits)):
        try: pfam = searchPfam(hits[i]).keys()[0]
        except:
            try:
                polymer = parsePDBHeader(hits[i], 'polymers')[0]
                dbref = polymer.dbrefs[0]
                pfam = searchPfam(dbref.accession).keys()[0]
            except: pfam = 'UNKNOWN'
        hits_pfam.append(pfam)
        if debug: pbar.update(1)
    if debug: pbar.close()

    # Split Hits Into Pos, Neg And Unk
    if debug: print("Splitting Hits By Pfam...")
    pos, neg, unk = [], [], []
    for i in range(len(hits)):
        pdb_id = hits[i]
        pfam = hits_pfam[i]
        if pfam in pos_pfam: pos.append(pdb_id)
        elif pfam == 'UNKNOWN': unk.append(pdb_id)
        else: neg.append(pdb_id)

    # Write Results To File
    if debug: print("Writing PDB Ids To File...")
    with open(pdb_folder + pdb_id_file + 'pos.csv', 'w') as f:
        for i in range(len(pos)): f.write(pos[i][:4]+','+pos[i][4]+'\n')
        print "Pos Hits Saved in:", f.name
    with open(pdb_folder + pdb_id_file + 'neg.csv', 'w') as f:
        for i in range(len(neg)): f.write(neg[i][:4]+','+neg[i][4]+'\n')
        print "Neg Hits Saved in:", f.name
    with open(pdb_folder + pdb_id_file + 'unk.csv', 'w') as f:
        for i in range(len(unk)): f.write(unk[i][:4]+','+unk[i][4]+'\n')
        print "Unk Hits Saved in:", f.name
    shutil.rmtree(pdb_folder+'temp')
