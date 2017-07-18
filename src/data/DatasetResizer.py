'''
FILENAME: DatasetResizer.py
INITIAL DATE: 18 July 2017
REVISED DATE: 18 July 2017

PROGRAM STATUS: Unfinished, non-operable. 
'''

import glob
import math
import os
import random as rd

def get_subdirectories(target_dir):
    qpaths = glob.glob(os.path.join(target_dir, '*/'))
    if len(qpaths) == 0:
        print('Error: no subdirectories found. Are you in the correct root folder?')
        exit(1)

    return qpaths

def get_dataset_size(num_gpus, num_data_elements):
    '''
    INPUTS:
        "num_gpus" is an integer, number of devices the dataset will be
            distributed across. 
        "num_data_elements" is an integer, number of discrete data 
            elements contained in the set you are looking at. 
    RETURNS:
        A datset size evenly divisible by the number of gpus via forced 
            integer division.
    '''
    return num_data_elements // num_gpus

def suggest_batch_sizes(train_set_size, test_set_size, num_gpus):
    suggest_list = []
    for i in range(1000):
        pbs = i * num_gpus
        if train_set_size % pbs == 0 and test_set_size % pbs == 0:
            suggest_list.append(pbs)
    suggest_list = suggest_list[:5]
    
    return suggest_list

def count_directory_contents(target_dir):
    return len([x for x in os.listdir(target_dir) if os.path.isfile(
        target_dir + x)])

# TODO: fix parser arg descriptions.
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
    basenames = [os.path.basename(p[:-1]) for p in os.listdir(os.getcwd())]
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

def main():
    
    # Need to sum contents of all subdirs under train and test 
    # Count subdirs under train and test
    # The total amount to remove from each dir is proportional to that dirs 
    # total size in the overall dataset.  Idea is to keep proportions of data set.
    #og_train_size
    #og_test_size
    target_dir = get_target_directory()
    qpaths = get_subdirectories(cwd)
    
    new_train_size =  get_dataset_size(num_gpus, og_train_size)
    print(get_dataset_size(7, 157695))
    suggested_batches = suggest_batch_sizes()
    print('Suggested batch sizes: {}'.format())
   
if __name__ == '__main__':
    main()
