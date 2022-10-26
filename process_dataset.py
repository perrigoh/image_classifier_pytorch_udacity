# /aipnd-project/process_dataset.py - Completed

# PROGRAMMER: Perri Goh
# DATE CREATED: 8 Oct 2022                                  
# REVISED DATE: 26 Oct 2022
# 


# Imports python modules
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from collections import OrderedDict

import json

from PIL import Image
import numpy


def tf_loader():
    """   
    Pre-process of image dataset for image classifier
      - Transform image: Image augmentation, resize and normalisation
      - Load image dataset into training, validation and test dataset.
    Parameters:
      None.
    Returns:
      1. train_data: Transformed image dataset for training 
      2. valid_data: Transformed image dataset for validation
      3. test_data: Transformed image dataset for testing
      4. trainloader: Load image dataset with a batch size of 64 images with shuffle for training
      5. validloader: Load image dataset with a batch size of 64 images for validation
      6. testloader: Load image dataset with a batch size of 64 images for testing.
      """           
    # specific data directory
    data_dir = 'flowers'
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'


    # transforms for the training, validation, and testing sets
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                        transforms.RandomResizedCrop(224),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])

    valid_transforms = transforms.Compose([transforms.Resize(255),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])

    test_transforms = transforms.Compose([transforms.Resize(255),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])
    # load the datasets with ImageFolder
    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    valid_data = datasets.ImageFolder(valid_dir, transform=valid_transforms)
    test_data = datasets.ImageFolder(test_dir, transform=test_transforms)

    # using the image datasets and the trainforms, define the dataloaders
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True)
    validloader = torch.utils.data.DataLoader(valid_data, batch_size=32)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=32)
    
    return train_data, valid_data, test_data, trainloader, validloader, testloader


def process_image(image_path):
    ''' 
    Scales, crops, and normalizes a PIL image for a PyTorch model,
    returns an Numpy array
    Parameters:
      1. image_path (str): The path of the image
    Returns:
      1. image: The transformed and normalised image for tensor.
    '''    
    # Process a PIL image for use in a PyTorch model
    pil_image = Image.open(image_path)
 
    image_transforms = transforms.Compose([transforms.Resize(256),
                                           transforms.CenterCrop(224)])                                    
                                       
    image = image_transforms(pil_image)
    image = numpy.array(image)/255
    
    means = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    
    image = (image - means) / std
    image = image.transpose((2,0,1))
  
    return image


# load cat_to_name.json for mapping categories to actual flowers name
def load_cat_name(category_names='cat_to_name.json'):
    """   
        Read a json file and store as variable cat_to_name
        Parameters:
          1. category_names (str): A json file that store the category names 
        Returns:
          1. cat_to_name: A dictionary with key pair values of indices and 
             category names.
    """
    with open(category_names, 'r') as f:
        cat_to_name = json.load(f)
    return cat_to_name

 