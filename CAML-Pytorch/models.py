import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#print(device)

"""LeNet5"""
class LeNet5(nn.Module):
    """
    [1] https://github.com/ChawDoe/LeNet5-MNIST-PyTorch/blob/master/model.py
    [2] https://github.com/erykml/medium_articles/blob/master/Computer%20Vision/lenet5_pytorch.ipynb 
    """
    def __init__(self, num_labels):
        super(LeNet5, self).__init__()
        n_acti = 5
        self.acti = [nn.ReLU() for i in range(n_acti)]
        
        self.conv1 = nn.Conv2d(3, 6, 3) # (28 - 3)/1 + 1 = 26
        self.pool1 = nn.MaxPool2d(2) # (26 - 2)/2 + 1 = 13 # AveragePool2D was used in Gradual_Domain_Adaptation
        self.conv2 = nn.Conv2d(6, 16, 2) # (13 - 2)/1 + 1 = 12
        self.pool2 = nn.MaxPool2d(2) # (12 - 2)/2 + 1 = 6 ######### !

        self.fc1 = nn.Linear(16*6*6, 120) 
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_labels)

    def forward(self, x):
        y = self.pool1(self.acti[0](self.conv1(x)))
        y = self.pool2(self.acti[1](self.conv2(y)))
        y = y.view(y.shape[0], -1)
        y = self.acti[2](self.fc1(y))
        y = self.acti[3](self.fc2(y))
        y = self.acti[4](self.fc3(y))
        return y