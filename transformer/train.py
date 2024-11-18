# train.py
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from tqdm import tqdm

import kornia.feature as KF

from vit_model import ViT  # Import the Vision Transformer model

from datasets import load_dataset



# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model and training parameters
IMG_SIZE = 64
PATCH_SIZE = 16
NUM_CLASSES = 10
EMBED_DIM = 768    # Potential problem: too many parameters
DEPTH = 12
NUM_HEADS = 12
MLP_RATIO = 4.0
BATCH_SIZE = 64
EPOCHS = 10
LEARNING_RATE = 1e-4
MAX_POINT_NUM = 64

##########################################################################################
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

# Data loading and preprocessing
# train_transforms = transforms.Compose([
#     transforms.RandomResizedCrop(64),
#     transforms.RandomHorizontalFlip(),
#     transforms.ToTensor(),
#     transforms.Normalize(mean, std),
# ])

# val_transforms = transforms.Compose([
#     transforms.Resize(64),
#     transforms.CenterCrop(64),
#     transforms.ToTensor(),
#     transforms.Normalize(mean, std),
# ])

# transform = transforms.Compose([
#     transforms.Resize((IMG_SIZE, IMG_SIZE)),
#     transforms.ToTensor(),
#     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
# ])

# train_dataset = ImageFolder(root='/projectnb/ec523kb/projects/teams_Fall_2024/Team_3/data', transform=train_transforms)
# val_dataset = ImageFolder(root='/projectnb/ec523kb/projects/teams_Fall_2024/Team_3/data', transform=val_transforms)

# TODO: problem: filter the binary images. Make sure RaFs correspond to the images. 

# train_dataset = load_dataset("zh-plus/tiny-imagenet", split='train').set_transform(train_transforms)
# val_dataset = load_dataset("zh-plus/tiny-imagenet", split='valid').set_transform(val_transforms)

# train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
# val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

###################################################################################################

# ds = load_dataset("zh-plus/tiny-imagenet")

# train_loader = DataLoader(ds['train'], batch_size=16, shuffle=True)
# val_loader = DataLoader(ds['valid'], batch_size=16, shuffle=False)


train_img = load_dataset2tensor('train.parquet')

# Initialize model, loss function, and optimizer
model = ViT(img_size=IMG_SIZE,
            patch_size=PATCH_SIZE,
            num_classes=NUM_CLASSES, 
            embed_dim=EMBED_DIM, 
            depth=DEPTH, 
            num_heads=NUM_HEADS, 
            max_point_num = MAX_POINT_NUM,
            mlp_ratio=MLP_RATIO,
            feature = KF.KeyNetAffNetHardNet,
            )

model = model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Training and evaluation functions (from the previous example)
def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for images, labels in tqdm(dataloader, desc="Training", leave=False):
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    epoch_loss = running_loss / len(dataloader)
    accuracy = 100. * correct / total
    return epoch_loss, accuracy

def evaluate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Evaluating", leave=False):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)  # Add lafs
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    epoch_loss = running_loss / len(dataloader)
    accuracy = 100. * correct / total
    return epoch_loss, accuracy

# Training loop
for epoch in range(EPOCHS):
    print(f"Epoch {epoch + 1}/{EPOCHS}")
    train_loss, train_accuracy = train_one_epoch(model, train_loader, criterion, optimizer, device)
    print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%")
    val_loss, val_accuracy = evaluate(model, val_loader, criterion, device)
    print(f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%\n")

print("Training complete!")
