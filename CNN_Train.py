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
# Import statemets

# Setting seet for randomizer 
seed = 67

np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Load Dataset Metadata
metadata_path = "./Garbage_data/Garbage_Dataset_Classification/metadata.csv"
metadata_df = pd.read_csv(metadata_path)


def get_file_path(metadata):
    image_folder = "./Garbage_data/Garbage_Dataset_Classification/images/"
    return os.path.join(image_folder, metadata["label"], metadata["filename"])


class GarbageDataset(Dataset):
    def __init__(self, metadata_df, transform=None):
        # Store metadata and optional image transformations
        self.metadata_df = metadata_df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        # Return total number of samples
        return len(self.metadata_df)

    def __getitem__(self, idx):
        # Fetch row for the given index
        metadata = self.metadata_df.iloc[idx]

        # Load and convert image from BGR (OpenCV) to RGB
        img_path = get_file_path(metadata)
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Get label string from metadata
        label = metadata["label"]

        # Apply transforms if provided
        if self.transform:
            image = self.transform(image)

        # Convert label to numerical index
        label_idx = label_to_idx[label]

        # Return image tensor and label index
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
        v2.RandomResizedCrop(size=(256, 256), antialias=True),
        v2.RandomPhotometricDistort(p=0.5),
        v2.RandomHorizontalFlip(p=0.5),
        v2.RandomVerticalFlip(p=0.5),
        v2.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ]
)

test_transform = v2.Compose(
    [
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Resize(256, antialias=True),
        v2.CenterCrop(256),
        v2.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ]
)

trainset = GarbageDataset(metadata_train_df, transform=transform)
testset = GarbageDataset(metadata_val_df, transform=test_transform)

trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=64, shuffle=True)
testloader = torch.utils.data.DataLoader(dataset=testset, batch_size=64, shuffle=False)

# change device (for mac use mainly)
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")

# Define LeNet Model, ultimately not used
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

# Define custom CNN
class CustomCNN(nn.Module):
    def __init__(self, num_classes=6):
        super(CustomCNN, self).__init__()

        # Convolutional feature extractor
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2),  # 128×128

            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2),  # 64×64

            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2),  # 32×32
        )

        # Classifier head
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 32 * 32, 256), nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        # Forward pass
        x = self.features(x)
        x = self.classifier(x)
        return x

class AxNet(nn.Module):
    def __init__(self, num_classes, dropout_rate=0):
        super(AxNet, self).__init__()
        self.num_classes = num_classes

        self.features = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=2),
            nn.BatchNorm2d(96),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),  # 63->31
            nn.Conv2d(96, 256, kernel_size=5, padding=2, stride=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),  # 31->15
            nn.Conv2d(256, 384, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(384),
            nn.ReLU(),  # 15-> 15
            nn.Conv2d(384, 384, kernel_size=3, padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU(),  # 15-> 15
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),  # 15-> 7
        )

        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(256 * 7 * 7, 1024),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

# Defining ensemble CNN to ensemble two models
class EnsembleCNN(nn.Module):
    def __init__(self, model1, model2):
        super(EnsembleCNN, self).__init__()
        self.model1 = model1
        self.model2 = model2

    def forward(self, x):
        output1 = self.model1(x)                   # Getting logits of both models
        output2 = self.model2(x)
        ensemble_output = (output1 + output2) / 2  # Averaging logits
        return ensemble_output


# Initialize both models
customcnn_model = CustomCNN(num_classes=6).to(device)
Ax_model = AxNet(num_classes=6, dropout_rate=0.15).to(device)

# Create the ensemble model
ensemble_model = EnsembleCNN(Ax_model, customcnn_model).to(device)


############ Choose Model To Train ##########################
model_to_train = ensemble_model
########################################################



# Loss and optimizer
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = optim.AdamW(model_to_train.parameters(), lr=0.0001, weight_decay=0.0001)
epochs = 100

# Calculate steps_per_epoch
steps_per_epoch = len(trainloader)

# Initialize the OneCycleLR scheduler
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=0.001,
    steps_per_epoch=steps_per_epoch,
    epochs=epochs,
    pct_start=0.3,
    div_factor=10.0,
    final_div_factor=25.0,
)

# Training loop
train_acc_list, test_acc_list = [], []
train_loss_list, test_loss_list = [], []
epoch_list = []

for epoch in range(epochs):
    model_to_train.train()
    running_loss, correct, total = 0.0, 0, 0

    for images, labels in trainloader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model_to_train(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        scheduler.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    train_loss = running_loss / len(trainloader)
    train_acc = 100 * correct / total

    # Evaluate model
    model_to_train.eval()
    test_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model_to_train(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    test_loss /= len(testloader)
    test_acc = 100 * correct / total

    print(
        f"Epoch [{epoch + 1}/{epochs}] -> Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%"
    )

    epoch_list.append(epoch + 1)
    train_loss_list.append(train_loss)
    test_loss_list.append(test_loss)
    train_acc_list.append(train_acc)
    test_acc_list.append(test_acc)


# moved for now
PATH = "Ensemble_model_states.pth"
torch.save(model_to_train.state_dict(), PATH)

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

# save model
# PATH = "Ax_model_states.pth"
# torch.save(Ax_model.state_dict(), PATH)

# load model using:
# loaded_model = AxNet()
# loaded_Ax_model.load_state_dict(torch.load(PATH))
