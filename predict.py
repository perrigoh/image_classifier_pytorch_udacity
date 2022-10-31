# /aipnd-project/predict.py Completed

# PROGRAMMER: Perri Goh
# DATE CREATED: 11 Oct 2022                                  
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

from get_args import get_args
from process_dataset import process_image
from process_dataset import load_cat_name

in_arg = get_args() 


def predict(checkpoint='checkpoint.pth', topk=3, category_names='cat_to_name.json'):
    """ 
    Predict the class (or classes) of an image using a trained deep learning model.
    Parameters:    
    1. save_dir (str): The directory where the model will be saved
    2. topk (int): The number highest ranking of the classes
    3. category_names (str): A json file that store the category names.      
    Returns:
    None - simply printing results.
    """    
    print('Loading model...')
    # load and rebuild model   
    
    # Use GPU if it's available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
          
    checkpoint = torch.load(checkpoint)
    input_layer = checkpoint['input_size']
    hidden_layer = checkpoint['hidden_layer']
    output_layer = checkpoint['output_size']
    arch = checkpoint['arch']                          
    model = models.__dict__[arch](pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
        model.classifier = nn.Sequential(nn.Linear(input_layer, hidden_layer),
                                        nn.ReLU(),
                                        nn.Dropout(p=0.2),
                                        nn.Linear(hidden_layer, output_layer),
                                        nn.LogSoftmax(dim=1))                                   
    model.load_state_dict(checkpoint['model_state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    
    model.to(device)
    # load cat_to_name.json for mapping categories to actual flowers name
    cat_to_name = load_cat_name(category_names)
            
    print('Predicting in process...')
    
    # Implement the code to predict the class from an image file
    model.eval()
    np_image = process_image(image_path='flowers/valid/11/image_03111.jpg')
    image = torch.from_numpy(numpy.array([np_image])).float()
    image = image.to(device)
    logps = model(image)

    ps = torch.exp(logps)
    top_p, top_class = ps.topk(topk)
    
    # .detach() detach computational graph in tensor
    # .tolist() return tensor value to standard python number as a nested list
    probs = top_p.cpu().detach().numpy().tolist()[0]
    classes = top_class.cpu().detach().numpy().tolist()[0]
        
    # Invert dictionary 
    idx_to_class = {v: k for k, v in model.class_to_idx.items()}

    # Mapping from index to class
    classes = [idx_to_class[c] for c in classes]
    cat_names = [cat_to_name[c] for c in classes]
    
    print('The topk {} prediction are:'.format(topk))
    print('Category names = {}'.format(cat_names))
    print('Classes = {}'.format(classes))
    print('Probabilites = {}'.format(probs))
    
    return

predict(in_arg.save_dir, in_arg.topk, in_arg.category_names)
