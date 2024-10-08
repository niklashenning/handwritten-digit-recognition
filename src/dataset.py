import os
import pandas as pd
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image


class HwD1000Dataset(Dataset):

    def __init__(self):
        # Init data frame and transformation settings
        self.dataframe = pd.read_csv('dataset/dataset.csv')
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((1,), (1,))
        ])

    def __len__(self):
        # Make sure len() can be used on objects of this class
        return len(self.dataframe)

    def __getitem__(self, idx):
        # Load image, convert it to grayscale, and apply transformation
        # before returning it along with its label
        image_path = os.path.join('dataset', self.dataframe.iloc[idx, 0])
        image = self.transform(Image.open(image_path).convert('L'))
        label = self.dataframe.iloc[idx, 1]
        return image, label
