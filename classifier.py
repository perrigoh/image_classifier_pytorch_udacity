# /aipnd-project/classifier.py - Completed

# PROGRAMMER: Perri Goh Meng Hsuan
# DATE CREATED: 11 Oct 2022                                  
# REVISED DATE: 
#

# Imports python modules
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from collections import OrderedDict


def classifier(arch, learning_rate, hidden_layer, gpu):
    """
    Build a image classification model using a pre-trained convolutional neural 
    network (CNN) with feature parameters are frozen. here are 3 options of 
    pre-trained model to choose from, Alexnet, VGG 16, VGG 13. The classifier 
    parameters are customised using nn.sequential a subclass of nn.module. Number
    of hidden layer = 1 , criterion = nn.NLLLoss (negative log likelihood loss),
    optimizer = optim.Adam. Activation function hidden layer = ReLU and output
    layer = Softmax.
    Parameters:    
    1. arch (str): The CNN architecture
    2. learning_rate(float): The a tuning parameter 
    3. hidden_layer (int): The number of hidden units
    4. gpu (str): The type of processing unit

    Returns:
    1. model: The built classifier model
    2. criterion: nn.NLLLoss
    3. optimizer: optim.Adam
    4. device: The type of processing unit
    """ 

    # processing unit selection
    if gpu == 'gpu':
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    
    # model selection
    if arch == 'alexnet':
        model = models.alexnet(pretrained=True)
        input_layer = 9216
        output_layer = 102

    elif arch == 'vgg16':
        model = models.vgg16(pretrained=True)
        input_layer = 25088
        output_layer = 102

    else:
        model = models.vgg13(pretrained=True)
        input_layer = 25088
        output_layer = 102

    # Freeze parameters so we don't backprop through them
    for param in model.parameters():
        param.requires_grad = False

        model.classifier = nn.Sequential(nn.Linear(input_layer, hidden_layer),
                                   nn.ReLU(),
                                   nn.Dropout(p=0.2),
                                   nn.Linear(hidden_layer, output_layer),
                                   nn.LogSoftmax(dim=1))                                   
          
    criterion = nn.NLLLoss()

    # Only train the classifier parameters, feature parameters are frozen
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)    
    
    model.to(device)
    
    return model, criterion, optimizer, device


