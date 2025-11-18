import os
import kaggle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
from torchvision.transforms import v2


# Load Dataset Metadata
metadata_path = "./Garbage_data/Garbage_Dataset_Classification/metadata.csv"
metadata_df = pd.read_csv(metadata_path)


def get_file_path(metadata):
    image_folder = "./Garbage_data/Garbage_Dataset_Classification/images/"
    return os.path.join(image_folder, metadata["label"], metadata["filename"])


class GarbageDataset(Dataset):
    def __init__(self, metadata_df, transform=None):
        self.metadata_df = metadata_df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.metadata_df)

    def __getitem__(self, idx):
        metadata = self.metadata_df.iloc[idx]
        img_path = get_file_path(metadata)
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        label = metadata["label"]

        if self.transform:
            image = self.transform(image)

        label_idx = label_to_idx[label]
        return image, label_idx


split_seed = 13

metadata_train_df, metadata_val_df = train_test_split(
    metadata_df, random_state=split_seed, stratify=metadata_df["label"].values
)

labels = sorted(metadata_df["label"].unique())
label_to_idx = {label: idx for idx, label in enumerate(labels)}

# Data Augmentation & Preprocessing
transform = v2.Compose(
    [
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.RandomResizedCrop(size=(32, 32), antialias=True),
        v2.RandomPhotometricDistort(p=0.5),
        v2.RandomHorizontalFlip(p=0.5),
        v2.RandomVerticalFlip(p=0.5),
        v2.RandomChannelPermutation(),
        v2.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ]
)

trainset = GarbageDataset(metadata_train_df, transform=transform)
testset = GarbageDataset(metadata_val_df, transform=transform)

trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=32, shuffle=True)
testloader = torch.utils.data.DataLoader(dataset=testset, batch_size=32, shuffle=False)

# change device (for mac use mainly)
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")


# Define LeNet Model
class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 5)
        self.fc1 = nn.Linear(64 * 5 * 5, 512)
        self.drop1 = nn.Dropout(0.6)
        self.fc2 = nn.Linear(512, 256)
        self.drop2 = nn.Dropout(0.6)
        self.fc3 = nn.Linear(256, 6)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.drop1(F.relu(self.fc1(x)))
        x = self.drop2(F.relu(self.fc2(x)))
        x = self.fc3(x)
        return x


# Modify the CustomCNN Class as per the request to integrate with the ensemble model
class CustomCNN(nn.Module):
    def __init__(self, num_classes=6):
        super(CustomCNN, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # -> 16x16
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # -> 8x8
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # -> 4x4
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# Define Ensemble Model to combine both LeNet and CustomCNN
class EnsembleCNN(nn.Module):
    def __init__(self, model1, model2):
        super(EnsembleCNN, self).__init__()
        self.model1 = model1
        self.model2 = model2

    def forward(self, x):
        output1 = self.model1(x)
        output2 = self.model2(x)
        ensemble_output = (output1 + output2) / 2  # Averaging logits
        return ensemble_output


# Initialize both models
lenet_model = LeNet().to(device)
customcnn_model = CustomCNN(num_classes=6).to(device)

# Create the ensemble model
ensemble_model = EnsembleCNN(lenet_model, customcnn_model).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss(label_smoothing=0.4)
optimizer = optim.AdamW(ensemble_model.parameters(), lr=0.0001, weight_decay=0.01)

# Training loop
epochs = 30
train_acc_list, test_acc_list = [], []
train_loss_list, test_loss_list = [], []
epoch_list = []

for epoch in range(epochs):
    ensemble_model.train()
    running_loss, correct, total = 0.0, 0, 0

    for images, labels in trainloader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = ensemble_model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    train_loss = running_loss / len(trainloader)
    train_acc = 100 * correct / total

    # Evaluate model
    ensemble_model.eval()
    test_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            outputs = ensemble_model(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    test_loss /= len(testloader)
    test_acc = 100 * correct / total

    print(
        f"Epoch [{epoch + 1}/{30}] -> Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%"
    )

    if (epoch + 1) % 5 == 0 or epoch == 30 - 1:
        epoch_list.append(epoch + 1)
        train_loss_list.append(train_loss)
        test_loss_list.append(test_loss)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)

# Plot Accuracy and Loss Graphs
plt.figure(figsize=(12, 5))

# Accuracy Plot
plt.subplot(1, 2, 1)
plt.plot(epoch_list, train_acc_list, "bo-", label="Train Accuracy")
plt.plot(epoch_list, test_acc_list, "ro-", label="Test Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy (%)")
plt.title("Train vs Test Accuracy")
plt.legend()

# Loss Plot
plt.subplot(1, 2, 2)
plt.plot(epoch_list, train_loss_list, "bo-", label="Train Loss")
plt.plot(epoch_list, test_loss_list, "ro-", label="Test Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Train vs Test Loss")
plt.legend()

plt.show()
