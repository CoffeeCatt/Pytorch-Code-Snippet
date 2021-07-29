from torch import nn  # type: ignore
import numpy as np  # type: ignore
import torch  # type: ignore
from torch.utils import data  # type: ignore
from numpy.random import RandomState  # type: ignore
from livelossplot import PlotLosses
import matplotlib.pyplot as plt


import time
import copy
import sys


def train_model(model, data_loaders, dataset_sizes, criterion, 
                optimizer, scheduler=None, num_epochs=25, model_path = None,
                device = 'cuda'):
    since = time.time()
    liveloss = PlotLosses()

    best_model_wts = copy.deepcopy(model.state_dict())
                        
    best_acc = 0.
    best_epoch = 0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()# Set model to training mode

            running_loss = 0.0
            running_corrects = 0

            
            for i, (inputs, labels) in enumerate(data_loaders[phase]):
                inputs = inputs.to(device)
                #inputs = inputs.view(-1, 28*28)
                labels = labels.to(device)

                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):  
                    #model Linear: (batch_size * input_size)--> (bs*output_size)
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels)
                print("\rIteration: {}/{}, Loss: {}.".format(
                    i+1, len(data_loaders[phase]), loss.item() * inputs.size(0)), end="")
                sys.stdout.flush()
                
                
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            if phase == 'train':
                avg_loss = epoch_loss
                t_acc = epoch_acc
            else:
                val_loss = epoch_loss
                val_acc = epoch_acc
            
            if phase == 'val' and epoch_acc> best_acc:
                best_model_wts = copy.deepcopy(model.state_dict())
                best_epoch = epoch
                best_acc = epoch_acc
                torch.save(model.state_dict(), model_path)
                
        liveloss.update({
            'accuracy': t_acc,
            'log loss': avg_loss,
            'val_accuracy': val_acc,
            'val_log loss': val_loss,
        })
                
        liveloss.draw()
        print('Train Loss: {:.4f} Acc: {:.4f}'.format(avg_loss, t_acc))
        print('Val Loss: {:.4f} Acc {:.4f} Best Acc: {:.4f}'.format(val_loss, val_acc, best_acc))
        print('Best epoch (acc): {}'.format(best_epoch))
        print()
        
        if epoch-best_epoch > 10:
            print('model stopped due to early stopping')
            break
    
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    
    # load best model weights
    model.load_state_dict(best_model_wts)
    print('Best Val acc: {:4f}'.format(best_acc))

    return model