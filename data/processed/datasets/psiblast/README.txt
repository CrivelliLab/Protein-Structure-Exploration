


NOTE: Below is now depricated but preserved for historical reasons. Now
sticking to 1, 2, 4, and 6 card runs to keep things even. Elements have been
moved back to their home directories to get the nice even splits that they were
at before. 

The HH-512-MS-FULL dataset has been modified in order to fit on 7 cards.
Originally it was of size:

train/ (total both classes): 142,500
    subtracted 15 files from positive class in order to make the total 142,485
    for even divsibiltiy by 21 (7 cards, 3 each per batch)

test/ (total both classes): 35,600
    subtracted 5 files from the positive class in order to make the total
    35,595 for even divisibility by 21 (same reasoning as above). 

These files were not deleted but were moved into their respective train/pos and
test/pos folders under ./SUBTRACTED_ELEMENTS_HH-512-MS-FULL so that the
original dataset can be rebuild with simple moves if needed. 

Move operations were accomplished via:

shuf -n 10 -e * | xargs -i mv {} path-to-new-folder
