# /aipnd-project/train.py - Completed

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

from time import time
import numpy

from get_args import get_args
from classifier import classifier
from process_dataset import tf_loader


def main():
    """ 
    Train the classifier model.
    Parameters:    
    1. arch (str): The CNN architecture
    2. learning_rate(float): The a tuning parameter 
    3. hidden_layer (int): The number of hidden units
    4. epochs (int): Number of iterations
    5. gpu (str): The type of processing unit     
    Returns:
    None - simply printing results.
    """
    # Measures total program runtime by collecting start time
    start_time = time()
    
    # retrieves command line arguments
    in_arg = get_args()
    
#   assign variable to user input arguments 
    epochs = in_arg.epochs
       
    # Build model
    print('Building model...')
    model, criterion, optimizer, device = classifier(in_arg.arch, in_arg.learning_rate, in_arg.hidden_units)
    
    # load transformed datasets
    print('Transforming datatsets...')
    train_data, valid_data, test_data, trainloader, validloader, testloader = tf_loader()

    # train model
    print('Training begin...') 
    for epoch in range(epochs=1):
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

# Call to main function to run the program
if __name__ == "__main__":
    main()



