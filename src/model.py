import torch
import torch.nn as nn


class Model(nn.Module):

    def __init__(self):
        super(Model, self).__init__()
        # Create two fully connected layers
        # The first layer takes an input of size 28 * 28 and outputs a tensor of size 128
        # The second layer takes the output of the first layer as an input and produces
        # an output of size 10 (the amount of classes in the dataset)
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        # Reshape (flatten) the input images and pass the reshaped input
        # through the fully connected layers and the ReLU activation function
        x = x.view(-1, 28 * 28)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x
