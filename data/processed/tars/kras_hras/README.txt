NOTE: KRAS_HRAS_24_JULY_FULL.tar.gz contains train/test subdirectories that have
already been split along 80/20 split levels. This dataset has also been pruned
down so that each split is divisible by 100 using bash ("ls -U | head -N | xargs
rm" from within each directory where N is the number of files to remove).

KRAS_HRAS_24_JULY_40K.tar.gz contains the same but chopped down to 40K total
elements. 
