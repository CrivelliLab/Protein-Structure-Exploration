*******************************************************************************
INITIAL DATE: 24 JULY 2017
REVISED DATE: 27 JULY 2017
DATASET GUIDE
*******************************************************************************

TODO:

    RENAME THE DIRECTORIES / CORRECT THE TrainModels CODE TO REFLECT!
    ADD IN MINI or MICRO?
    CHANGE TO BIG MED SMALL?

NAVIGATION NOTE:

    There are two directories of note: datasets/ and tars/. datasets/ contains the
    actual segmented and prepared data directories that can be used to train a
    network using Keras' "flow from directory" functionality, while tars/ contains
    the tarfiles used to build the directories in datasets in case they need to be
    analyzed or rebuilt.

METHODOLOGY: 

    For all datasets, shuffling and subsetting images into split train/test
    directories was accomplished via InteractiveDataSubsetter.py using an 80/20
    train/test split. All dataset size pruning was accomplished by running the bash
    command "ls -U | head -NUMBER | xargs rm" from within the dirctory to be
    pruned. -NUMBER is an integer value indicating the number of files to be
    removed from the working directory. 

datasets/

    1) KRAS_HRAS_BIG: 
        Generated on 24 July 2017 . The number of train and test elements were
        pruned down to reach sizes evenly divisible by 100 to facilitate easy
        training via Keras' "flow from directory" functionality. Note that all
        KRAS_HRAS datasets have been pruned so that they only include proteins
        that have a diameter less than or equal to 64 angstroms in order to
        ensure consistant 2D encoding richness.

        train/
            |
            KRAS/: 29,400 images
            |
            HRAS/: 62,600 images

        test/
            |
            KRAS/: 7,300 images
            |
            HRAS/: 15,600 images
    
    2) KRAS_HRAS_SMALL:
        Generated on 26 July 2017 by taking the segmented and shuffled
        HRAS_KRAS_BIG dataset and pruning each training class size to 16,000
        elements and each test class size to 4,000 elements to acheive a total
        dataset size of 40,000 elements (20,000 per class).

        train/
            |
            KRAS/: 16,000 images
            |
            HRAS/: 16,000 images

        test/
            |
            KRAS/: 4,000 images
            |
            HRAS/: 4,000 images
        
    3) PSIBLAST_BIG:
    4) PSIBLAST_SMALL:
    5) RAS-WD40_BIG:
    6) RAS-WD40_SMALL:


tars/
