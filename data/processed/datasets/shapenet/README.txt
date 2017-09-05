This dataset was prepared on 10 August 2017.

The subdirectories were generated according to a roughly 75 / 25 train / test
split. Furthermore, they have been pruned down so that each of the two splits
are even numbers to allow for batch sizes of 50 or 25 elements when training. 

EncodedShapeNetVox32/ contains two subdirectoreis, train/ and test/

    train/ contains 35,100 elements

    test/ contains 8,750 elements


SUBTRACTED_ELEMENTS_EncodedShapeNetVox32/ contains two subdirectories,

    train/ contains one subdorectory, 
    
        04379243/, which contains 18 elements. This is the class from which
                    elements were taken (it was the biggest one).

    test/ contains one subdirectory with the same name, conatining 0 elements. 
            This is because nothing had to be pruned from the original test
            split to make it divisible by 50 and 25. 

To rebuild the OG dataset, just move the elements of 

    SUBTRACTED_ELEMENTS_EncodedShapeNetVox32/train/04379243/

back to their original location. 
