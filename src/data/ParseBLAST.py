'''
ParseBlast.py
Auhtor: Rafael Zamora
Updated: 07/11/17

README:

The following script is used to parse Psi-Blast search results and return a list
of unique ids for protein structures in the Protein Data Bank.

Global variables used to parse the Blast results are defined under #- Global variables.

'''
import os, shutil, argparse
from prody import searchPfam, pathPDBFolder, parsePDBHeader
from tqdm import tqdm

#- Global Variables
blast_results = 'P89Z6VMZ015-Alignment-HitTable.csv'
positive_uniprot = ['P01111', 'P01112', 'P01116']

#- Verbose Settings
debug = True

################################################################################

if __name__ == '__main__':

    # Cmd Line Args
    parser = argparse.ArgumentParser()
    parser.add_argument('-br', '--blast_results',
                        help="Blast Hit Table Results CSV",
                        type=str, default=None)
    parser.add_argument('-pu', '--positive_uniprot',
                        help="UniProt IDs for Positive Cases; comma seperated values",
                        type=str, default=None)
    args = vars(parser.parse_args())
    if args['blast_results']: blast_results = args['blast_results']
    if args['positive_uniprot']: positive_uniprot = args['positive_uniprot'].split(',')

    # File Paths
    os.chdir(os.path.dirname(os.path.realpath(__file__)))
    blast_results = '../../data/raw/BLAST/' + blast_results
    pdb_folder = '../../data/raw/PDB/'
    pdb_id_file = '-'.join(positive_uniprot)
    if not os.path.exists(pdb_folder+'temp'): os.makedirs(pdb_folder+'temp')
    pathPDBFolder(pdb_folder+'temp')

    # Read Blast Results
    if debug: print("Parsing Blast Results...")
    hits = []
    with open(blast_results) as f:
        for line in f.readlines():
            cols = line.split(',')
            if len(cols) > 2:
                pdb_id = cols[1].split('|')[3]
                hits.append(pdb_id)
    hits = list(set(hits)) # remove duplicates

    # Get Set Of Positive Families
    if debug:
        print("Fetching Postive Pfam Ids...")
        pbar = tqdm(total=len(positive_uniprot))
    pos_pfam = []
    for i in range(len(positive_uniprot)):
        pfam = searchPfam(positive_uniprot[i]).keys()[0]
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
    pos = []
    neg = []
    unk = []
    for i in range(len(hits)):
        pdb_id = hits[i]
        pfam = hits_pfam[i]
        if pfam in pos_pfam: pos.append(pdb_id)
        elif pfam == 'UNKNOWN': unk.append(pdb_id)
        else: neg.append(pdb_id)

    # Write Results To File
    if debug: print("Writing PDB Ids To File...")
    with open(pdb_folder + pdb_id_file + '-pos_ids.txt', 'w') as f:
        for i in range(len(pos)): f.write(pos[i]+'\n')
    with open(pdb_folder + pdb_id_file + '-neg_ids.txt', 'w') as f:
        for i in range(len(neg)): f.write(neg[i]+'\n')
    with open(pdb_folder + pdb_id_file + '-unk_ids.txt', 'w') as f:
        for i in range(len(unk)): f.write(unk[i]+'\n')
    shutil.rmtree(pdb_folder+'temp')
