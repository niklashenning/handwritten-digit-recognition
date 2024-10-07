import torch
import torch.nn as nn
import pandas as pd
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image


class HwD1000Dataset(Dataset):

    def __init__(self):
        self.dataframe = pd.read_csv('dataset/dataset.csv')
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((1,), (1,))
        ])

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        image_path = 'dataset/' + self.dataframe.iloc[idx, 0]
        image = self.transform(Image.open(image_path).convert('L'))
        label = self.dataframe.iloc[idx, 1]
        return image, label


class Model(nn.Module):

    def __init__(self):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x
