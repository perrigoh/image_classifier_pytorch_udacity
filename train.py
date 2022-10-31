# /aipnd-project/train.py - Completed

# PROGRAMMER: Perri Goh
# DATE CREATED: 11 Oct 2022                                  
# REVISED DATE: 28 Oct 2022
# 


# Imports python modules
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from collections import OrderedDict

from time import time
import numpy

from get_args import get_args
from process_dataset import tf_loader

in_arg = get_args() 


def train(arch='vgg13', hidden_layer=512, learning_rate=0.01, epochs=1):
    """
    Build a image classification model using a pre-trained convolutional neural 
    network (CNN) with feature parameters are frozen. here are 3 options of 
    pre-trained model to choose from, Alexnet, VGG 16, VGG 13. Train the classifier 
    using customised parameters. The nn.sequential is a subclass of nn.module. Number
    of hidden layer = 1 , criterion = nn.NLLLoss (negative log likelihood loss),
    optimizer = optim.Adam. Activation function hidden layer = ReLU and output
    layer = Softmax. This function will print out the train and validation loss,
    followed by saving the model to rebuild for prediction or further training. 
    Parameters:    
    1. arch (str): The CNN architecture with default value 'vgg13' 
    2. hidden_layer (int): The number of hidden units with default value '512'
    3. learning_rate (float): The a tuning parameter with default value '0.01'
    4. epochs (int): Number of iterations with default value '1'.
    
    Returns:
    None - simply printing results.
    """

    # Measures total program runtime by collecting start time
    start_time = time()
     
    # Build model
    print('Building model...')

    # Use GPU if it's available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # model selection
    if arch == 'alexnet':
        model = models.__dict__[arch](pretrained=True)
        input_layer = 9216
        output_layer = 102

    elif arch == 'vgg16':
        model = models.__dict__[arch](pretrained=True)
        input_layer = 25088
        output_layer = 102

    elif arch == 'vgg13':
        model = models.__dict__[arch](pretrained=True)
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
    
      
    # load transformed datasets
    print('Transforming datatsets...')
    train_data, valid_data, test_data, trainloader, validloader, testloader = tf_loader()

    # train model
    print('Training begin...') 
    for epoch in range(epochs):
        for inputs, labels in trainloader:
            # Move input and label tensors to the default device
            inputs, labels = inputs.to(device), labels.to(device)
            # Set previous computed gradient (weight & biase) to 0
            # so that new computed gradient will not add on to the 
            # old which been updated in the previous loop
            optimizer.zero_grad()
            logps = model(inputs)
            loss = criterion(logps, labels)
            train_loss = loss.item()

            loss.backward()
            optimizer.step()

        else:
            # deactivate the dropout layer
            model.eval()
            train_accuracy = 0
            # turn off the computation of gradient (weight, bias)
            with torch.no_grad():
                for inputs, labels in validloader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    logps = model(inputs)
                    loss = criterion(logps, labels)
                    # .item() return tensor value to standard python number
                    valid_loss = loss.item()

                    # calculate accuracy
                    ps = torch.exp(logps)
                    top_ps, top_class = ps.topk(1, dim=1)
                    equality = top_class == labels.view(*top_class.shape)
                    train_accuracy += torch.mean(equality.type(torch.FloatTensor)).item()

                    print('Epoch: {}/{}'.format(epoch +1, epochs))
                    print('Train Loss: {:4f}, Valid Loss: {:4f}, Train Accuracy: {:4f}'.format(train_loss, valid_loss,
                        train_accuracy/len(validloader)))                       

            model.train()  
    print('End of training...')

    # save the trained model to checkpoint
    print('Saving model checkpoint...')
    model.class_to_idx = train_data.class_to_idx
    
    checkpoint = {'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'criterion_state_dict': criterion.state_dict(),
                'input_size': model.classifier[0].in_features,
                'output_size': model.classifier[3].out_features,           
                'hidden_layer': model.classifier[0].out_features,
                'arch': arch,
                'epochs': epochs,
                'class_to_idx': model.class_to_idx}

    torch.save(checkpoint, 'checkpoint.pth')
    
    print('Done!')
    
    # measure total program runtime by collecting end time
    end_time = time()
    
    # computes overall runtime in seconds & prints it in hh:mm:ss format
    tot_time = end_time - start_time #calculate difference between end time and start time
    print("\n** Total Elapsed Runtime:",
          str(int((tot_time/3600)))+":"+str(int((tot_time%3600)/60))+":"
          +str(int((tot_time%3600)%60)) )

    return

train(in_arg.arch, in_arg.hidden_units, in_arg.learning_rate, in_arg.epochs)

