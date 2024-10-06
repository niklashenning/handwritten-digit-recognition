import pandas as pd
from torch.utils.data import Dataset
from PIL import Image


class HwD1000Dataset(Dataset):

    def __init__(self):
        self.dataframe = pd.read_csv('dataset/dataset.csv')

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        image_path = 'dataset/' + self.dataframe.iloc[idx, 0]
        image = Image.open(image_path)
        label = self.dataframe.iloc[idx, 1]
        return image, label
