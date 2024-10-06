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
