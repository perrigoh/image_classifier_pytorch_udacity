#!/usr/bin/env python3
#-----------------------50-----------------------|----------72---------|--79--|

# Imports python module
import argparse

def get_args():

    """
    Retrieves and parses the command line arguments provided by the user
    when they run the programme from a terminal window. This function uses 
    Python's argparse module to create and define command line arguments. 
    If the user provides a flag without argument, the const value will be
    used as the argument. If the user fails to provide some or all the 
    arguments, then the default values are used for the missing arguments.
    Command Line Arguments:
    1. Directory to save checkpoints as --save_dir with default value 
       'checkpoint.pth'
    2. Model Architecture as --arch with default value 'vgg13'
    Hyperparameters:
    3. Learning Rate as --learning_rate with default value 0.01
    4. Hidden Units as --hidden_units  with default value 512
    5. Epochs as --epochs with default value 1
    6. Top K as --top_k  with default value 3
    7. Mapping of categories to real names as --category_names with default 
       value 'cat_to_name.json'
    This function returns these arguments as an ArgumentParser object.
    Parameters:
     None - simply using argparse module to create & store command line arguments
    Returns:
     parse_args() - data structure that stores the command line arguments object.  
    """

    # Create Parser using ArgumentParser
    parser = argparse.ArgumentParser()

    # arguments for train.py
    # Option: python train.py data_dir --save_dir save_directory
    parser.add_argument('--save_dir', dest='save_dir', nargs='?', type=str,
                        const='checkpoint.pth', default='checkpoint.pth', 
                        help='Set directory to save checkpoint') 
    # Option: python train.py data_dir --arch
    parser.add_argument('--arch', dest='arch', nargs='?', type=str,
                        const='vgg13', default='vgg13', 
                        help='Choose architecture, alexnet, vgg13 or vgg16')
    # Option: python train.py data_dir --learning_rate --hidden_units --epochs
    parser.add_argument('--learning_rate', dest='learning_rate', nargs='?',
                        type=float, const=0.01, default=0.01,
                        help='Set hyperparameters - learning rate type=float eg. 0.01')
    parser.add_argument('--hidden_units', dest='hidden_units', nargs='?',
                        type = int, const=512, default=512,
                        help='Set hyperparameters - hidden_units type=int range 102 - 9216')
    parser.add_argument('--epochs', dest='epochs', nargs='?', type=int, 
                        const=1, default=1,
                        help='Set hyperparameters - epochs')

    # arguments for predict.py
    # Option: python predict.py input checkpoint --topk
    parser.add_argument('--topk', dest='topk', nargs='?', type=int,
                        const=3, default=3, 
                        help = 'Top K most likely classes')
    # Option: python predict.py input checkpoint --category_names cat_to_name.json
    parser.add_argument('--category_names', dest='category_names', nargs='?',
                        type=str, const='cat_to_name.json', default='cat_to_name.json', 
                        help='A list of category names in json format')

    return parser.parse_args()