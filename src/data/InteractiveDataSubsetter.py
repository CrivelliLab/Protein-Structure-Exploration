'''
FILE NAME: InteractiveDataSubsetter.py

INITIAL DATE: 8 July 2017
REVISED DATE: 17 July 2017

PROGRAM STATUS: Core functionality tested and operational. Significant 
    clunkiness & TODOs. High probability of bugs being present. 

PROGRAM USEAGE:

    Case 1) To operate on data files / folders contained in the current working 
        directory enter 'python3 data_subsetter.py' at the command line in the directory
        containing this file and follow the prompts that appear. 
    Case 2) To operate on data files / folders contained in an arbitrary
        directory (not necessarily the current directory) enter 
        'python3 data_subsetter.py -td absolute/path/to/target/directory' at the 
        command line and then follow the prompts that appear. If the location
        of the class data you wish to segment and / or serialize is a
        subdirectory of the directory this program is invoked from, you may
        omit the absolute path name of the target directory and instead simply
        use the subdirectory's name. 

    HINT: 
    
        If you are buried/in/a/very/deep/directory/structure and need to
        operate this program on a subdirectory contained within your current
        working directory, a shortcut to writing out the absolute path is to type
        $PWD/subdirectory and bash will expand the $PWD variable out to the full
        path for you. 

DETAILED PROGRAM DESCRIPTION:

    This is an interactive module designed to allow for easy command-line
    subsetting and serialization of image datasets for neural network training. The
    motivation for this program is the difficulty and repetition associated with
    preparing datasets for ingestion by neural networks and other machine learning
    systems. This module is specifically intended to segment and serialize image
    datasets. 

    This program looks at the target directory for an arbitrary number of subdirectories,
    each of which is assumed to contain images belonging to a single class, (i.e.,
    a single type of image like trees or Ras proteins). For each subdirectory / class,
    the module shuffles and segments the class into various split types depending
    upon the user's preference. 

NOTE: 

    This module has only been tested under Python 3K. Minimal changes would
    get it to work under Python 2 but no effort has been made to make it
    compatible at this time. It has also only been designed to run under *nix
    systems, although it *may* work with some modification on MacOS. 

NON-CRITICAL TODO:
    A) Create a nice conda env list for running this preprocessor (and ideally
        a list for running all of the distributed network on Cori but
        especially this preprocessing module).
    B) Output a txt file describing the splits and what happened. 
    C) Pretty print all terminal output using shutil, pprint, and textwrap. 
        See:
        https://stackoverflow.com/questions/37572837/how-can-i-make-python-3s-print-fit-the-size-of-the-command-prompt
    D) Add tqdm progress meters. 
    E) Add ability to generate small datasets (i.e. a percent of total dataset
        size) from the total image set for small-scale testing. 

CRITICAL TODO:
    A) Append split information and class labels to a timestamped information
        file. 
    B) Validate that the format is CIFAR10 compatible. 
    C) Add a seed to the random image shuffler to be able to reproduce the same
        datasets if needed? Might not be necessary. 
        
'''
# *****************************************************************************
# Import Statements
# *****************************************************************************
import argparse as ap
import cv2
from collections import defaultdict
import glob
import h5py
import math
import numpy as np
import os
import random as rd
import shutil
from subprocess import call
import sys
import tarfile
import time
from tqdm import tqdm

# *****************************************************************************
# FUNCTION DEFINITIONS
# *****************************************************************************
def create_directories(path, dir_names):
    '''
    DESCRIPTION: 
        Creates len(dir_names) number of directories at the 
        location provided in path.
    INPUTS: 
        "path" is an absolute path to the directory in which the
            subdirectories are to be created. 
        "dir_names" is the list of subdirectory names and corresponds to 
            the selected mode. I.e., dir_names will contain "train", "test", 
            and / or "validation".
    RETURNS: 
        None, generates specified subdirectories if successful. Returns
        OsError and exits on failure. 
    '''
    if len(dir_names) > 0:
        print('Attempting to create {} new subdirectories {}...'.format(
            len(dir_names), dir_names))
        for dir_name in dir_names:
            try:
                os.makedirs(os.path.join(path, dir_name))
            except OSError:
                print('Error creating directory {}. Exiting.'.format(
                    os.path.join(path, dir_name)))
                exit(1)
    return 

# TODO: Parallelize this if it isn't too much of a pain. 
# TODO: Prepare for splitting into stand-alone module for separate,
# non-interactive use. Allow it to accept path information as argument. This
# will require updating the doc string as well. 
# TODO: add argument to allow for generating a limited subset of the total
# dataset for use on a weaker machine / for a faster run. This could be fixed
# at 10% of size, or maybe be a flexible param. The user inputs this before
# running, or does this module prompt them for a percent after indexing the
# contents? Prolly the latter...
# TODO: tqdm display.
def split_classes(splits_dict):
    '''
    DESCRIPTION: 
        Splits a dataset into a number of subdirectories equal to the number of
        keys in splits_dict, which is assumed to be either 2 or 3 corresponding
        to a data segementation scheme of train/test or train/test/validation.
        Alternative data segmentation approaches will require modification of
        this code. 

        Each final subdirectory will have a name corresponding to one of the
        keys in splits_dict, and will contain a number of its own
        subdirectories equal to the number of classes (i.e. data-containing
        directories) in the top-level directory this program is operating in. 
        
        For example, if this program is run in a directory containing two
        subdirectories, one named 'goblins' that contains 1000
        images of goblins, and the other named 'wizards' that contains 800
        images of wizards, and splits_dict contains three key/value pairs
        {train:70, test:20, validation:10}, then this module will generate the
        folllowing output heirarchy:
                
        /top_level_dir/
                    |
                    |-> train/
                    |       |
                    |       |-> goblins/
                    |       |       ...700 images 
                    |       |
                    |       |-> wizards/
                    |               ...560 images
                    |
                    |-> test/                    
                    |       |
                    |       |-> goblins/
                    |       |       ...200 images
                    |       |
                    |       |-> wizards/
                    |               ...160 images
                    |
                    |-> validation/
                                |
                                |-> goblins/
                                |       ...100 images
                                |
                                |-> wizards/
                                        ...80 images

    INPUTS:
        "splits_dict", a dictionary containing 2 or three key/value pairs
        depending on user's segmentation choice (i.e. train/test or
        train/test/validation segmentation schemes). The keys of the dict are
        expected to be the name of the split, e.g. 'train' or 'test', and are
        used to create the subdirectory structure when segmentation occurs. 
    RETURNS:
        None. Generates the desired subdirectory strucutre with all training 
    '''
    cwd = os.getcwd()
    
    # Get dataset split information. 
    train_s = splits_dict['train']
    test_s = splits_dict['test']
    if len(splits_dict) == 3:
        valid_s = splits_dict['validation']
    
    # Dict keys are split names, will be directory names. I.e. 'train', 'test'.
    split_dirs = [key for key in splits_dict.keys()]
    
    print('Searching current working directory for subdirectories...')
    qpaths = get_subdirectories()
    print('{} subdirectories found:'.format(len(qpaths)))
    for qpath in qpaths:
        print(os.path.basename(qpath[:-1]))
    create_directories(cwd, split_dirs)
    print('Done')
   
    # Loop over subdirectories, which are assumed to contain one class each.
    # Count subdirectory contents, determine number of train/test/validation
    # elements for the class.
    for qpath in qpaths:
        basename = os.path.basename(qpath[:-1]) # Removes trailing '/'
        print('Entering subdirectory {}'.format(str(basename)))
        print('Counting files in directory. This may take a moment...')
        filename_list = [x for x in os.listdir(qpath) if os.path.isfile(qpath +
            x)]
        num_dir_contents = len(filename_list)
        train_num = int(train_s * num_dir_contents)
        test_num = int(math.floor(test_s * num_dir_contents))
        validation_num = num_dir_contents - (train_num + test_num)
        print('{} files found'.format(num_dir_contents))
        print('Shuffling files...')
        rd.shuffle(filename_list)
        print('Splitting directory with the following file counts:')
        
        # For the case where only a train/test split is desired. 
        if len(splits_dict) == 2:
            print('{}/train/: {}'.format(basename, train_num))
            print('{}/test/: {}'.format(basename, test_num))
            for split_dir in split_dirs:
                if split_dir == 'train': split = train_num
                if split_dir == 'test': split = test_num
                try:
                    os.makedirs(os.path.join(split_dir, basename))
                    for f in filename_list[:split]:
                        shutil.move(os.path.join(qpath, f), os.path.join(
                            split_dir, basename))
                    filename_list = filename_list[split:]
                except OSError:
                    print('Error. Exiting.')
                    exit(1)
            print('{} subdirectory successfully segmented'.format(basename))

        # For the case where a train/test/validation split is desired. 
        if len(splits_dict) == 3:
            print('{}/train/: {}'.format(basename, train_num))
            print('{}/test/: {}'.format(basename, test_num))
            print('{}/validation/: {}'.format(basename, validation_num))
            for split_dir in split_dirs:
                if split_dir == 'train': split = train_num
                if split_dir == 'test': split = test_num
                if split_dir == 'validation': split = validation_num
                try:
                    os.makedirs(os.path.join(split_dir, basename))
                    for f in filename_list[:split]:
                        shutil.move(os.path.join(qpath, f), os.path.join(
                            split_dir, basename))
                    filename_list = filename_list[split:]
                except OSError:
                    print('Error. Exiting.')
                    exit(1)
            print('{} subdirectory successfully segmented'.format(basename))

    # Removing the directories the images were moved out of that are now empty. 
    print('Removing intermediate directories.')
    for qpath in qpaths:
        if not os.listdir(qpath):
            try:
                print('Deleting directory: {}'.format(qpath))
                os.rmdir(qpath)
            except OSError:
                print('Error: cannot remove directory. Ignoring.')
        else:
            print('Error: {} appears to contain files. Skipping.'.format(qpath))

    return

# TODO: Segement into own module file for separate, non-interactive use. Need
# to allow it to accept path info as an argument. 
def create_hdf5(splits_dict, images_shape_dict, sep_hdf5s_flag):
    '''
    DESCRIPTION:
        Create a binary version of a dataset.
        NOTE: HDF5 ships with a variety of different low-level drivers, which map the logical
        HDF5 address space to different storage mechanisms. You can specify which
        dirver you want to use when the file is opened. E.g., the HDF5 "core" driver
        can be used to create a purely in-memory HDF5 file, optionally written out to
        disk when it is closed. DEFALUT IS NONE.
    INPUTS: 
        "splits_dict" is a dict containing either 2 or 3 entries
            depending upon whether the user has requested a train/test split or
            a train/test/validation split. The keys are expected to be 'train'
            'test' and optionally 'validation'. All values must be integers. 
        "images_shape" is a dict containing the
            keys "image_heights", "image_widths", and "image_depths", which are
            expected to be integers. "image_depth" describes the number of channels 
            present in the image data, e.g. 1 for monochrome and 3 for RGB. 
        "sep_hdf5s_flag" is a boolean flag. If True then this program will
            generate a separate .hdf5 file for each split in splits_dict (e.g. if
            splits_dict contains {train:70 test:20 validation:10} then a True
            sep_hdf5s_flag will result in three .hdf5 files: train.hdf5,
            test.hdf5, and validation.hdf5. If sep_hdf5s_flag is False then all
            three splits are processed into a single .hdf5 file containing
            subgroups 'train', 'test', and 'validation'.  
    RETURNS:
        None. Generates the requested .h5 files or raises error. 
    '''
    # Helper fxn creates hdf5 file for writing using input path.
    def create_hdf5_file(hdf5_path):
        return h5py.File(hdf5_path, mode='w')

    # Helper fxn creates nested dictionaries. 
    def nested_dict():
        return defaultdict(nested_dict)

    # Get split informaton from the incoming dict. 
    train_s = splits_dict['train']
    test_s = splits_dict['test']
    if len(splits_dict) == 3:
        valid_s = splits_dict['validation']
    
    # Get list of subdirectories containing class data.
    # Make sure to ignore train/test/validation subdirs that may have been 
    # generated by this program previously in directory segmentation mode. 
    qpaths = get_subdirectories()
    for qpath in qpaths:
        if os.path.basename(qpath[:-1]) in ['train', 'test', 'validation']:
            qpaths.remove(qpath)
   
    # Dir basename used for dataset name if sep_hdf5s_flag is False. 
    if not sep_hdf5s_flag:
        single_file_name = (os.path.basename(os.getcwd()) + '.hdf5') 

    # Initialize data structures for image, label, and slice info. 
    addrs_combined = [] # List of all image addresses for all classes. 
    labels_combined = [] # List of all image labels for all classes. 
    #data_slices = {} # Dict contains slices of class dataset. For sinlge hdf5. 
    slice_dicts = nested_dict() # Nested dict for building hdf5 files. 

    # Get all class images and labels, shuffle them, append to combined lists.
    for i, qpath in enumerate(qpaths):
        basename = os.path.basename(qpath[:-1]) # Removes trailing '/'.
        class_addrs = glob.glob(os.path.join(qpath, '*.png'))
        class_labels = [i for addr in class_addrs]
        # TODO: append to a .txt file each class name's corresponding ID. 

        # Combine data, shuffle, separate. 
        class_addr_and_labels = list(zip(class_addrs, class_labels))
        rd.shuffle(class_addr_and_labels)
        class_addrs, class_labels = zip(*class_addr_and_labels)
        
        # Append the current class data to the combined data addrs and labels lists. 
        addrs_combined.extend(class_addrs)
        labels_combined.extend(class_labels)

    # Divide the dataset into train/test/validation splits as per splits_dict.
    # TODO: combine the two blocks, just have the if / else
    if len(splits_dict) == 2: # i.e. only a train/test split.
        train_end = int(train_s * len(addrs_combined))
        train_addrs = addrs_combined[:train_end]
        train_labels = labels_combined[:train_end]

        test_addrs = addrs_combined[train_end:]
        test_labels = labels_combined[train_end:]
        
        if sep_hdf5s_flag:
            # Build the slice_dicts dict with two nested dicts. 
            slice_dicts['train'] = {}
            slice_dicts['test'] = {}

            slice_dicts['train']['slice_data'] = train_addrs
            slice_dicts['train']['slice_labels'] = train_labels
            slice_dicts['test']['slice_data'] = test_addrs
            slice_dicts['test']['slice_labels'] = test_labels
        else:
            # Build the slice_dicts dict with one nested dict. 
            slice_dicts[single_file_name] = {}
            
            slice_dicts[single_file_name]['train_data'] = train_addrs
            slice_dicts[single_file_name]['test_data'] = test_addrs

    if len(splits_dict) == 3: # i.e. a train/test/validation split. 
        train_end = int(train_s * len(addrs_combined))
        train_addrs = addrs_combined[:train_end]
        train_labels = labels_combined[:train_end]

        test_end = train_end + int(test_s * len(addrs_combined))
        test_addrs = addrs_combined[train_end:test_end]
        test_labels = labels_combined[train_end:test_end]

        val_addrs = addrs_combined[test_end:]
        val_labels = labels_combined[test_end:]
        
        if sep_hdf5s_flag:
            # Build the slice_dicts dict with three nested dicts. 
            slice_dicts['train'] = {}
            slice_dicts['test'] = {}
            slice_dicts['validation'] = {}

            slice_dicts['train']['slice_data'] = train_addrs
            slice_dicts['train']['slice_labels'] = train_labels
            slice_dicts['test']['slice_data'] = test_addrs
            slice_dicts['test']['slice_labels'] = test_labels
            slice_dicts['validation']['slice_data'] = val_addrs
            slice_dicts['validation']['slice_labels'] = val_labels
        else:
            # Build the slice_dicts dict with one nested dict. 
            slice_dicts[single_file_name] = {}
            
            slice_dicts[single_file_name]['train_data'] = train_addrs
            slice_dicts[single_file_name]['test_data'] = test_addrs
            slice_dicts[single_file_name]['validation_data'] = val_addrs
       
    # Get image shape data, load as elements in slice_dicts{}.
    img_h = images_shape_dict['image_heights']
    img_w = images_shape_dict['image_widths']
    img_d = images_shape_dict['image_depths']

    # Next create image and label datasets according to entries in slice_dicts{}.
    # Separate files case first. 
    if sep_hdf5s_flag:
        print('Creating datasests in separate .hdf5 files')

        # Getting the input data shapes. 
        slice_dicts['train']['slice_data_shape'] = (len(train_addrs), 
                img_h, img_w, img_d) # Tensorflow likes this 'NHWC' format. 
        slice_dicts['test']['slice_data_shape'] = (len(test_addrs),
                img_h, img_w, img_d) 
        if len(splits_dict) == 3:
            slice_dicts['validation']['slice_data_shape'] = (len(val_addrs),
                    img_h, img_w, img_d) 
        
        # Programmer's Note: type(slice_dicts) = ,class 'collections.defaultdict'>
        # Programmer's Note: type(slice_dicts[slice_dict]) = <class 'dict'>
        for slice_dict in slice_dicts: # E.g., 'train', 'test', 'validation'.
            slice_name = str(slice_dict)
            slice_data = slice_dicts[slice_dict]['slice_data']
            slice_data_shape = slice_dicts[slice_dict]['slice_data_shape']
            slice_labels = slice_dicts[slice_dict]['slice_labels']
            dataset_name = slice_name + '.hdf5' # e.g. "train.hdf5"
            hdf5_path = os.path.join(os.getcwd(), dataset_name)
            hdf5_file = create_hdf5_file(hdf5_path)
            # Create image dataset for slice. 
            hdf5_file.create_dataset('data', slice_data_shape, np.int8)
            # Create label dataset for slice and fill it now. 
            hdf5_file.create_dataset('labels', (len(slice_labels),), np.int8)
            hdf5_file['labels'][...] = slice_labels
            print('Writing images to {}. This will take some time!'.format(
                dataset_name))
            # Read images one by one and save to the hdf5. 
            # Convert images to img_h X img_w, cv2 loads images as BGR, so we convert
            # to RGB if img_d > 1.
            for i in tqdm(range(len(slice_data))):
                addr = slice_data[i] # address of an image in a data slice.
                img = cv2.imread(addr)
                img = cv2.resize(img, (img_h, img_w),
                        interpolation=cv2.INTER_CUBIC)
                if img_d > 1: # i.e. if images are not monochrome.
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                hdf5_file['data'][i,...] = img[None] # Save image.
            print('All files written to {}'.format(dataset_name))
            print('Closing {}'.format(dataset_name))
            hdf5_file.close() 
        print('All datasets successfully written.')
    
    # Single dataset file case. 
    else:
        '''
        slice_dicts[single_file_name]['train_data_shape'] = (len(train_addrs),
                img_h, img_w, img_d)
        slice_dicts[single_file_name]['test_data_shape'] = (len(test_addrs),
                img_h, img_w, img_d)
        if len(splits_dict) == 3:
            slice_dicts[single_file_name]['validation_data_shape'] = (len(val_addrs), 
                    img_h, img_w, img_d)
        '''
        train_data_shape = (len(train_addrs), img_h, img_w, img_d)
        test_data_shape = (len(test_addrs), img_h, img_w, img_d)
        if len(splits_dict) == 3:
            validation_data_shape = (len(val_addrs), img_h, img_w, img_d)
        
        # TODO: cwd()
        hdf5_path = os.path.join(os.getcwd(), single_file_name)
        # Create single hdf5 file.
        hdf5_file = create_hdf5_file(hdf5_path)

        # Create images datasets.
        print('Creating datasets for single file...')
        hdf5_file.create_dataset('train_data', train_data_shape, np.int8)
        hdf5_file.create_dataset('test_data', test_data_shape, np.int8)
        if len(splits_dict) == 3:
            hdf5_file.create_dataset('validation_data', validation_data_shape, np.int8)
        
        # Create labels datasets. #TODO: move up into same location as labels
        # for separate files case. 
        hdf5_file.create_dataset('train_labels', (len(train_labels),), np.int8)
        hdf5_file['train_labels'][...] = train_labels
        hdf5_file.create_dataset('test_labels', (len(test_labels),), np.int8)
        hdf5_file['test_labels'][...] = test_labels
        if len(splits_dict) == 3:
            hdf5_file.create_dataset('validation_labels', (len(val_labels),), np.int8)
            hdf5_file['validation_labels'][...] = val_labels
        print('Done')

        # Read images one by one and save to the hdf5. 
        # Convert images to img_h X img_w, cv2 loads images as BGR, so we convert
        # to RGB if img_d > 1.
        # TODO: progres meter. 
        print('Writing images to datasets. This will take some time!')
        for split_name, data  in slice_dicts[single_file_name].items():
            print('Writing images to {} dataset...'.format(split_name))
            for i in tqdm(range(len(data))):
                addr = data[i] # address of an image in a data slice.
                img = cv2.imread(addr)
                img = cv2.resize(img, (img_h, img_w),
                        interpolation=cv2.INTER_CUBIC)
                if img_d > 1: # i.e. if images are not monochrome.
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                hdf5_file[split_name][i,...] = img[None] # Save image.
            print('All files written to {} dataset'.format(split_name))
        print('All datasets successfully written to {}. Closing .hdf5.'.format(
            single_file_name))
        hdf5_file.close() 

    return

def get_subdirectories():
    '''
    DESCRIPTION: Gets the absolte paths of all subdirectories contained within
        the current working directory.
    INPUTS: None. Functions off of the CWD. 
    RETURNS: a list of absolute paths to all subdirectories in the CWD.
    '''
    qpaths = glob.glob(os.path.join(os.getcwd(),'*/'))
    if len(qpaths) == 0:
        print('Error: no subdirectories found. Are you in the correct root folder?')
        exit(1)
    
    return qpaths

def confirm(prompt, resp=False):
    '''
    DESCRIPTION: Simple confirmation prompt validator. 
    INPUTS: A string prompt requesting a binary yes / no response. 
    RETURNS: True or False depending on user input. 
    '''
    prompt = '%s [%s]|%s: ' % (prompt, 'n', 'y')
    while True:
        ans = input(prompt)
        if not ans:
            return resp
        if ans not in ['y', 'Y', 'n', 'N']:
            print('Please enter "y" or "n"')
            continue
        if ans == 'y' or ans == 'Y':
            return True
        if ans == 'n' or ans == 'N':
            return False

def untar_directories():
    # TODO: fix tqdm display.
    '''
    DESCRIPTION: untars all ".tar.gz" files found in the current directory
        before segmentation or serialization processes begin. 
    INPUTS: None. 
    RETURNS: None. Decompresses all .tar.gzs in CWD. 
    '''
    print('Untaring all archives in directory. This may take some time...')
    try:
        i = 0
        num_tars = len([x for x in os.listdir(os.getcwd()) if x.endswith('.tar.gz')])
        for file in tqdm(os.listdir(os.getcwd()), total = num_tars):
            if file.endswith('.tar.gz'):
                tar = tarfile.open(file)
                tar.extractall()
                tar.close()
                i += 1
    except:
        e = sys.exc_info()[0]
        print('Error: {}'.format(e))
        exit(1)
    print('{} archives successfully decompressed'.format(i))
    
    return

def get_inputs():
    '''
    DESCRIPTION: 
        Purely for convenience! Runs a simple interactive prompt to get run params. 
        "sd" mode is used to split images into subdirectories by class and 
        train/test/validation split for use in a Keras-style "flow_from_directory" 
        data loading scheme. "hdf5" mode creates a serialized (binary) .h5 file 
        for distributed network training on a large dataset. The binary data format 
        is desireable for very large datasets and distributed networks where data IO 
        bottlenecks are a concern. 
    INPUTS: None.
    RETURNS: A variable-length dict of 4 - 7 key/value pairs depending on whether the
        selected split type is train/test ("tt") or train/test/validation
        ("ttv") and what segmentation mode is slected. The dict contains the split 
        proportions for the given mode, the image shape information if the mode 
        happens to be "hdf5", directory before beginning segmentation. The whole 
        thing is probably overkill.
    '''
    # Initializing defaults for vars, return dict. 
    inputs_dict = {}
    mode = None
    sep_hdf5s_flag = False
    tar_flag = False
    
    # Input prompts for interactive use. 
    # TODO: might want to add a mode for untarring vs. subdirectories.
    # TODO: remove tar backup option??
    split_type_prompt =  ('Enter "tt" for train/test split or "ttv" ' 
            'for train/test/validation split: ')
    train_split_prompt = ('Enter the percent of data you wish to use for '
            'training as an integer: ')
    test_split_prompt = ('Enter the percent of data you wish to use for ' 
            'testing as an integer: ')
    validation_split_prompt = ('Enter the percent of data you wish to use for '
            'validation as an integer: ')
    mode_prompt = ('If you want to split classes into subdirectories enter '
            '"sd" otherwise enter "hdf5" to serialize all data into binary '
            '.h5 files. Both options segment data according to the split '
            'mode previously selected: ')
    separate_hdf5s_prompt = ('Generate separate hdf5 files for each dataset '
            'segment, or create one hdf5 file containing subgroups for each '
            'dataset segment? Enter "s" for separate or "o" for one: ')
    img_height_prompt = ('Enter the height in pixels of the input images as '
            'an integer: ')
    img_width_prompt = ('Enter the width in pixels of the input images as '
            'an integer: ')
    img_depth_prompt = ('Enter the depth (i.e. number of channels) of the '
            'input images as an integer (e.g. 1 for monochrome, 3 for RGB): ')
    tar_prompt = ('If the data you wish to segment are in compressed '
            '".tar.gz" archives the must be decompressed before segmentation '
            'can occur. Decompress? Enter "y" or "n": ')

    while True:
        num_splits = input(split_type_prompt)
        if num_splits.lower() not in ['tt', 'ttv']:
            print('Please enter "tt" or "ttv"')
            continue
        # Two splits.
        if num_splits.lower() == 'tt':
            try:
                train_split = int(input(train_split_prompt))
            except ValueError:
                print('Not an integer')
                continue
            try:
                test_split = int(input(test_split_prompt))
            except ValueError:
                print('Not an integer')
                continue
            if train_split + test_split != 100:
                print('Split proportions must sum to 100%')
                continue
            inputs_dict['train'] = train_split/100.0
            inputs_dict['test'] = test_split/100.0
            break
        # Three splits.
        if num_splits.lower() == 'ttv':
            try:
                train_split = int(input(train_split_prompt))
            except ValueError:
                print('Not an integer')
                continue
            try:
                test_split = int(input(test_split_prompt))
            except ValueError:
                print('Not an integer')
                continue
            try:
                validation_split = int(input(validation_split_prompt))
            except ValueError:
                print('Not an integer')
                continue
            if train_split + test_split + validation_split != 100:
                print('Split proportions must sum to 100%')
                continue
            inputs_dict['train'] = train_split/100.0
            inputs_dict['test'] = test_split/100.0
            inputs_dict['validation'] = validation_split/100.0
            break

    while True:
        mode = input(mode_prompt)
        if mode.lower() not in ['sd', 'hdf5']:
            print('Please enter "sd" or "hdf5"')
            continue
        inputs_dict['mode'] = mode.lower()
        # Image shape information. Only needed in hdf5 mode. 
        if mode.lower() == 'hdf5':
            while True:
                try:
                    img_height = int(input(img_height_prompt))
                except ValueError:
                    print('Not an integer')
                    continue
                try:
                    img_width = int(input(img_width_prompt))
                except ValueError:
                    print('Not an integer')
                    continue
                try:
                    img_depth = int(input(img_depth_prompt))
                except ValueError:
                    print('Not an integer')
                    continue
                inputs_dict['image_heights'] = img_height
                inputs_dict['image_widths'] = img_width
                inputs_dict['image_depths'] = img_depth
                break

            while True:
                separate_hdf5s = input(separate_hdf5s_prompt)
                if separate_hdf5s.lower() not in ['s', 'o']:
                    print('Please enter "s" or "o"')
                    continue
                if separate_hdf5s.lower() == 's':
                    inputs_dict['sep_hdf5s_flag'] = True
                else:
                    inputs_dict['sep_hdf5s_flag'] = False
                break
        break

    while True:
        tar_flag = input(tar_prompt)
        if tar_flag.lower() not in ['y', 'n']:
            print('Please enter "y" or "n"')
            continue
        inputs_dict['tar_flag'] = tar_flag.lower()
        break

    return inputs_dict

def get_target_directory():
    '''
    DESCRIPTION:
        Fxn checks if the user supplied a target directory when invoking the 
            program. It will accept subdirectory names without an absolute
            path, but cannot handle relative pathing otherwise. 
    INPUTS:
        None. Argparse takes command-line-fed arguments. It is expected that
            the user will enter the absolute path to the directory containing 
            their classes (which themselves are expected to be in separate 
            subdirectories or archive files).  
    RETURNS:
        The directory that this program will operate on.
    '''
    parser = ap.ArgumentParser(description='An interactive module designed to '
            'guide a user through common data subsetting and formatting tasks '
            'associated with neural network training.')
    parser.add_argument('-td', '--target_directory', help=('The absolute path '
            'to the directory containing the data you wish to segment and / '
            'or serialize. This directory is expected to contain either '
            'separate "tar.gz" files for each class, or directories '
            'containing one class type per directory. The directory should '
            'not contain urealted subdirectories or acrchive files. If the '
            'target directory is a subdirectory of the current working '
            'directory (where this program is being run from) then it is '
            'acceptable to enter only the name of the directory without '
            'supplying the absolute path. DEFAULT: current working '
            'directory.'), nargs='?', type=str, default=os.getcwd())
    args = vars(parser.parse_args())
    w_dir = args['target_directory']
    # Check if input path name is a directory in the cwd.
    basenames = [os.path.basename(p[:-1]) for p in get_subdirectories()]
    if w_dir[:-1] in basenames:
        try:
            w_dir = os.path.join(os.getcwd(), w_dir)
            os.chdir(w_dir)
        except:
            e = sys.exc_info()[0]
            print('Error: {}'.format(e))
            exit(1)
    
    elif os.getcwd() != w_dir:
        try:
            os.chdir(w_dir)
        except:
            e = sys.exc_info()[0]
            print('Error: {}'.format(e))
            exit(1)
    else:
        assert w_dir == os.getcwd()
        
    return w_dir

# *****************************************************************************
# Program Execution
# *****************************************************************************
def main():
    '''
    DESCRIPTION:
        Takes a user through the full process of selecting a method of dataset
        segmentation (splitting into directories or serializing into .h5
        files according to user-provided segmentation percentages). This provides 
        only a narrow range of operations with the hope that it may speed up 
        common, repetitive tasks in data preparation for neural network training. 
    INPUTS: 
        "path" is a string, the path to the directory where the data you wish
            to operate on is located. 
        
        Additionaly, this module calls get_inputs() in order to source 
        parameters for segmentaion and serialization functions. 
    RETURNS:
        None. Generates either a segmented-by-folder dataset or a serialized
        .h5 datatset for neural network ingestion and / or other machine
        learning tasks. 
    ''' 
    w_dir = get_target_directory()
    print('Program operating on directory: {}'.format(w_dir))

    # Primary image data attributes useful for nerual net ingestion. 
    image_data_atrributes = ['image_heights', 'image_widths', 'image_depths']
    # Common machine learning data split types, extend as needed. 
    dataset_split_parameters = ['train', 'test', 'validation']

    print('Beginning parameter input phase. You will have the chance to '
            'confirm your choices before operations begin.')
    inputs_dict = get_inputs()

    # Flags.
    mode = inputs_dict['mode']
    sep_hdf5s_flag = inputs_dict['sep_hdf5s_flag']
    tar_flag = inputs_dict['tar_flag']

    # Separating shape data and split percents into separate dicts. 
    images_shape_dict = {x:inputs_dict[x] for x in image_data_atrributes if x
            in inputs_dict.keys()}
    splits_dict = {x:inputs_dict[x] for x in dataset_split_parameters if x in
            inputs_dict.keys()}

    # Confirmation prompts, segmentation / seraialization execution, and
    # tar.decompression
    if tar_flag == 'y':
        tar_prompt = ('CONFIRM: untar all *.tar.gz files in {} ? This may be a '
                'lengthy process.'.format(w_dir))
        if confirm(tar_prompt):
            untar_directories()
        else:
            print('Canceling archive extraction.')
    
    if mode == 'sd':
        seg_prompt = ('CONFIRM: alter current directory strucutre into {} '
                'subdirectories?'.format(len(splits_dict)))
        if confirm(seg_prompt):
            split_classes(splits_dict)
        else:
            print('Aborting')

    if mode == 'hdf5':
        if sep_hdf5s_flag:
            hd5_prompt = ('CONFIRM: serialize data into {} separate .hdf5 files?'.format(
                len(splits_dict)))
            if confirm(hd5_prompt):
                create_hdf5(splits_dict, images_shape_dict, sep_hdf5s_flag)
            else:
                print('Aborting')
        else:
            hd5_prompt = ('CONFIRM: serialize data into 1 .hdf5 file?')
            if confirm(hd5_prompt):
                create_hdf5(splits_dict, images_shape_dict, sep_hdf5s_flag)
            else:
                print('Aborting')

    if mode not in ['sd', 'hdf5']: # Execution should never reach here. 
        print('Unknown error. Exiting.')
        exit(1)

    print('Program complete. Exiting.')
    exit(0)


if __name__ == '__main__':
    main()
