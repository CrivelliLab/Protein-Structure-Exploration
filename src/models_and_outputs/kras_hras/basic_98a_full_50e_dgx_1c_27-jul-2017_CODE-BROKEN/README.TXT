NOTES: KRAS/HRAS full set, 1 Tesla P100 card, 32 filters per layer, no batch
normalization. 

UPDATE: 27 July 2017

This network is now our best-performing model. At epoch 43/100 it is performing
at 98.37% validation accuracy. Note that the class split on this dataset is
apporoximately 68/32 so baseline, coin-toss levels is 68%. 


INITIAL NOTES: 26 July 2017

Trying to run this one again after fixing pathing issue mentioned below. We may
not have gotten the real results from this network because I didn't delete the
.pyc files before launching a new network after modification. Need to look into
the details of this. 

this training run initially failed due to wacky pathing issues where the network couldnt
find the place to save the weights file after epoch 5. This was because this
net was set to draw data from a directory named "KRAS_HRAS_24_JULY_BIG" and to
save to a folder named "../../models/KRAS_HRAS_24_JULY_BIG_NO_BN/weights.hdf5"
but the saving of the file was tied to the OG base path name. It's because it
was trying to draw off the first folder and save to the second but the program
code didn't account for the base directory name difference. Needs to be fixed
to run properly.
