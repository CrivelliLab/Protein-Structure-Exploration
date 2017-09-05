UPDATED 31 July 2017

Unfortunately due to time constraints, the methods used to run networks, store
the results of network runs, and label directories have all changed over time
as the process has scaled. Efforts have been made to get everything into
alignment, and where code versions are not in alginment efforts have been made
to clearly mark the broken components. 

Notes on subdirectory contents:

Each of the subdirectories in this directory contain experiment output
corresponding to a dataset type, e.g. PSIBlast postiive/negative hits, Kras vs.
Hras classification, and the original generic RAS vs. WD40 repeat group datasets. 
Within each of the top-level dataset type directories there are a variable
number of experiment folders that are named according to the following
convention:
    
network-info_percent-accuracy_dataset-size_epochs-run_hardware-used_num-cores_date_optional-note

so for example in kras_hras/ there is a subfolder:

    basic_99a_full_336e_dgx_4c_27-jul-2017

and this means that the basic (12 conv layer, 1 dense layer, 64 filters, 3x3
kernel) network architecture was used, the model hit 99% accuracy on the full
dataset size (dataset information is available in the /data/processed/datasets/
directory from the root project directory), it reached this accuracy over 336
epochs, the DGX-1 was used, and 4 cards of the the DGX-1 were employed, and the
network was run on 27-july-2017. There is no optional note appended to this run
because it completed successfully, the code isn't currently broken (i.e. it can
be rerun if needed), it isn't currenlty training, and there are no other
notable items about the experiment status. If an experiment is currently
running it will have "_IN-PROGRESS" appended to the end of its directory name.
For example: 

    basic_99a_full_336e_dgx_4c_27-jul-2017_IN-PROGRESS

would mean that network is still training and should not be touched. 

This is an attempt at a self-documenting directory naming structure. It can be
improved upon, but a big thing to note is that in the beginning there was not
independent directory structure for separate test runs and that was added
later. Individual readmes should be appended to each experiment folder to
explain results, other notes, and problems in more detail than can be included
in the directory naming scheme. 

The following folders contain code that is deprecated and broken. This code was
used to generate the model outputs in these folders but was done before the
massive directory restructuring took place. In order to reproduce the network
runs that relied on the code in these folders the python scripts located in the
network definitions subdirectories will have to be modified. If that is desired
then the better option would be to use a new version of the network files and
simply update the pameters as needed. 

DIRECTORIES CONTAINING BROKEN CODE HAVE "CODE-BROKEN" APPENDED TO THEIR NAME. 

kras_hras/basic_90a_40k_128e_dgx_4c_26-jul-2017_CODE-BROKEN
kras_hras/basic_98a_full_50e_dgx_1c_27-jul-2017_CODE-BROKEN
psiblast/basic_87a_full_40e_dgx_4c_26-jul-2017_CODE-BROKEN
ras_wd40/basic_95a_full_100e_dgx_7c_24-july-2017_CODE-BROKEN
