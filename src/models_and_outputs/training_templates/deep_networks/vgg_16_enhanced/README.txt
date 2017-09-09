UPDATE FROM 8-SEPT-17

These network definitions are updated as of 8 Sept 2017 and may be used to run
networks. The VGG_ENHANCED-ORIGINAL.py is the original 18 million parameter
network used on the psiblast test that attained 98.5% accuracy. It was a mess
of half relu / selu activations. Only 1024 nodes in 2 dense layers, and 64
filters per layer. The new VGG_16_ENHANCED.py is the best current deep
architecture that we have. 12 Layers, 2 dense with 2048 nodes, a pattern of
filters per layer of 64-64-128-128-256-256-512-512-512-512-2048-2048-softmax.
Upping the number of dense layer nodes to 4096 gets a resource exhaustion
error.

Currently running as of 8-aug-2017.
The experiment is located under:
vgg16E_??a_full-augs-separate_150e_dgx_4c_8-sep-2017

***
OLD
***

The below network hit 98.5% accuracy over 100 epochs, with a messy network. It
is now interred at psiblast/vgg16E_98a_full_100e_dgx_2c_9-aug-2017

UPDATE FROM 9-AUG-17

These network definitions are updated as of 9 August 2017 and can be used to
run networks. In particular VGG_16_ENHANCED.py is updated to run and is
bug-free. On 9 August 2017 these files were used to run the experiment
contained within the directory:

        psiblast/vgg16E_??a_full_100e_dgx_2c_9-aug-2017_IN-PROGRESS
