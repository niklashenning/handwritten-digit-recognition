import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset, random_split
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


dataset = HwD1000Dataset()

training_size = int(0.8 * len(dataset))
validation_size = len(dataset) - training_size
training_data, validation_data = random_split(dataset, [training_size, validation_size])
training_dataloader = DataLoader(training_data, batch_size=32, shuffle=True)
validation_dataloader = DataLoader(validation_data, batch_size=32, shuffle=False)

model = Model()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

epochs = 10

for epoch in range(epochs):
    for images, labels in training_dataloader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    print('Epoch [{}/{}], Loss: {:.6f}'.format(epoch + 1, epochs, loss.item()))


model.eval()
correct = 0

with torch.no_grad():
    for images, labels in validation_dataloader:
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / validation_size
print('Test Accuracy: {:.2f}% ({}/{})'.format(accuracy, correct, validation_size))
