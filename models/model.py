"""
Description: This module defines a simple feedforward neural network model using PyTorch.

Author: Nisal Hemadasa
Date: 18-10-2024
Version: 1.0
"""
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        # Define the layers for MNIST dataset (28 x 28 dimensional images)
        self.dense1 = nn.Linear(in_features=28 * 28, out_features=10)  # Dense layer with 28 * 28 inputs and 10 outputs
        self.relu = nn.ReLU()  # ReLU activation function
        self.dense2 = nn.Linear(10, 1)  # Output layer with 10 inputs and 1 output
        # self.sigmoid = nn.Sigmoid()  # Sigmoid activation function

    def forward(self, x):
        """Forward pass through the network"""
        # Flatten the input to (batch_size, 784)
        x = x.view(x.size(0), -1)
        x = self.dense1(x)
        x = self.relu(x)
        x = self.dense2(x)
        # x = self.sigmoid(x)
        return x


# for MNIST and F_MNIST dataset (28 x 28 dimensional images)
class CNNMNIST(nn.Module):
    def __init__(self):
        """Defines the architecture of the network"""
        super(CNNMNIST, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)
        self.fc1 = nn.Linear(in_features=64 * 5 * 5, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=10)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        """Describes how input data flows through the network"""
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, kernel_size=2, stride=2)
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, kernel_size=2, stride=2)
        x = x.view(-1, 64 * 5 * 5)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        # return x
        return torch.log_softmax(x, dim=1)


# for CIFAR 10 dataset (32 x 32 dimensional images) in 3 channels
class CNNCIFAR10(nn.Module):
    """
    A Convolutional Neural Network for CIFAR-10 dataset.
    Input: 32x32 RGB images (3 channels)
    Output: 10-class classification
    """

    def __init__(self):
        super(CNNCIFAR10, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)

        # Fully connected layers
        self.fc1 = nn.Linear(in_features=128 * 4 * 4, out_features=256)  # Adjusted for CIFAR-10 size
        self.fc2 = nn.Linear(in_features=256, out_features=10)  # 10 classes for CIFAR-10

        # Dropout for regularization
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        # Convolutional and pooling layers
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)  # Output size: [batch_size, 32, 16, 16]

        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)  # Output size: [batch_size, 64, 8, 8]

        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)  # Output size: [batch_size, 128, 4, 4]

        # Flatten
        x = x.view(x.size(0), -1)  # Flatten to [batch_size, 128 * 4 * 4]

        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        # Log-Softmax for classification
        return F.log_softmax(x, dim=1)

# class CNNCIFAR10(nn.Module):
#     def __init__(self):
#         super(CNNCIFAR10, self).__init__()
#         self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3)
#         self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)
#         self.fc1 = nn.Linear(in_features=64 * 5 * 5, out_features=128)
#         self.fc2 = nn.Linear(in_features=128, out_features=10)
#         self.dropout = nn.Dropout(p=0.5)
#
#     def forward(self, x):
#         x = torch.relu(self.conv1(x))
#         x = torch.max_pool2d(x, kernel_size=2, stride=2)
#         x = torch.relu(self.conv2(x))
#         x = torch.max_pool2d(x, kernel_size=2, stride=2)
#         x = x.view(x.size(0), -1)
#         x = torch.relu(self.fc1(x))
#         x = self.dropout(x)
#         x = self.fc2(x)
#         # return x
#         return torch.log_softmax(x, dim=1)


def train(_model: nn.Module, _dataloader: DataLoader, epochs: int, verbose=False) -> None:
    """
    Train the network on the training set.
    :param _model: The model to train
    :param _dataloader: The dataloader containing training dataset
    :param epochs: The number of epochs to train for
    :param verbose: Whether to print training progress
    :return: None
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # criterion = nn.BCEWithLogitsLoss()
    # criterion = nn.BCELoss()
    criterion = nn.CrossEntropyLoss()
    _optimizer = torch.optim.Adam(_model.parameters(), lr=0.001)
    _model.train()
    _model.to(device)
    for epoch in range(epochs):
        correct, total, epoch_loss = 0, 0, 0.0
        # this loop is added because _dataset is dictionary like and torch.from_numpy() expects only Dataloader types.
        # Also takes batches of data from the dataset and trains the model

        # Define the maximum number of batches to use per epoch
        max_batches_per_epoch = 10

        # Limit the loop to a fixed number of batches
        for batch_idx, (_x, _y) in enumerate(_dataloader):
            if batch_idx >= max_batches_per_epoch:
                break

            # inputs = _x.unsqueeze(1).float()  # Ensure images are in the right format and shape to feed to the model
            inputs = _x.to(device, non_blocking=True)
            labels = _y.to(device, non_blocking=True).long()

            # Clear gradients for each batch
            _optimizer.zero_grad()

            # Forward pass
            outputs = _model(inputs)

            # Calculate loss
            _loss = criterion(outputs, labels)

            # Backward pass and optimization
            _loss.backward()
            _optimizer.step()

            # Metrics
            epoch_loss += _loss.item()
            total += labels.size(0)
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()

        epoch_loss /= len(_dataloader)
        epoch_acc = correct / total
        if verbose:
            print(f"Train Epoch {epoch + 1}: train loss {epoch_loss}, accuracy {epoch_acc}")


def test(_model: nn.Module, _dataset: DataLoader) -> Tuple[float, float]:
    """
    Evaluate the network on the entire test set.
    :param _model: The model to evaluate
    :param _dataset: The test dataset
    :return: Tuple of loss and accuracy
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # criterion = nn.BCEWithLogitsLoss()
    criterion = nn.CrossEntropyLoss()
    correct, total, loss = 0, 0, 0.0
    _model.eval()
    _model.to(device)
    with torch.no_grad():
        # this loop is added because _dataset is dictionary like and torch.from_numpy() expects only Dataloader types
        for _x, _y in _dataset:
            # inputs = _x.unsqueeze(1).float()   # Ensure images are in the right format and shape to feed to the model
            inputs = _x.to(device, non_blocking=True)
            labels = _y.to(device, non_blocking=True).long()  # CE needs Long targets

            # forward pass
            outputs = _model(inputs)

            # compute loss
            loss += criterion(outputs, labels).item()

            # compute accuracy
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    # average loss over all samples
    loss /= len(_dataset)
    accuracy = correct / total
    return loss, accuracy
