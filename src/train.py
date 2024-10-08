import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split
from model import Model
from dataset import HwD1000Dataset


# Init dataset
dataset = HwD1000Dataset()

# Split dataset into training data (80%) and validation data (20%)
# and create data loaders
training_size = int(0.8 * len(dataset))
validation_size = len(dataset) - training_size
training_data, validation_data = random_split(dataset, [training_size, validation_size])
training_dataloader = DataLoader(training_data, batch_size=32, shuffle=True)
validation_dataloader = DataLoader(validation_data, batch_size=32, shuffle=False)

# Init model, criterion, and optimizer
model = Model()
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=0.001)

# Training loop
epochs = 50
losses = []

for epoch in range(epochs):
    for images, labels in training_dataloader:
        optimizer.zero_grad()  # Reset the gradients of the model parameters to zero
        outputs = model(images)  # Forward pass (process input images and make predictions)
        loss = criterion(outputs, labels)  # Calculate loss (difference between prediction and label)
        loss.backward()  # Backward pass (compute the gradients)
        optimizer.step()  # Optimizer step (update the model parameters based on the gradients)

    losses.append(loss.item())
    print('Epoch [{}/{}], Loss: {:.6f}'.format(epoch + 1, epochs, loss.item()))


# Calculate accuracy of the model on the validation data
model.eval()
correct = 0

with torch.no_grad():
    for images, labels in validation_dataloader:
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / validation_size
print('Test Accuracy: {:.2f}% ({}/{})'.format(accuracy, correct, validation_size))

# Visualize training results
plt.get_current_fig_manager().set_window_title('Training')
plt.plot(range(epochs), losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()
