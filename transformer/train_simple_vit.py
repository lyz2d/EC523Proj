import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from datasets import load_dataset
from torch.optim import AdamW
import pytorch_lightning as pl

from simple_vit import SimpleViT
import os


# Hyperparameters
BATCH_SIZE = 64
IMAGE_SIZE = 160  # Assume square images for simplicity
PATCH_SIZE = 16
NUM_CLASSES = 100  # ImageNet-100 has 100 classes
DIM = 512
DEPTH = 6
HEADS = 8
MLP_DIM = 1024
LR = 3e-4
EPOCHS = 90
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Data Preparation
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.Lambda(lambda img: img.convert("RGB")),  # Ensure everything converted to 3 channels
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Load dataset and split
using_scc  = True
#If using scc to train it, 
if using_scc:
    dataset = load_dataset("ilee0022/ImageNet100", cache_dir='/projectnb/ec523kb/projects/teams_Fall_2024/Team_3/hg_cache')
else:
    dataset = load_dataset("ilee0022/ImageNet100")

train_data = dataset['train']
test_data = dataset['validation']

# Custom Dataset to apply transforms
class CustomImageDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        image = self.transform(sample['image'])
        label = sample['label']
        return image, label

# DataLoader
train_loader = DataLoader(CustomImageDataset(train_data, transform), batch_size=BATCH_SIZE, shuffle=True, num_workers=31)
test_loader = DataLoader(CustomImageDataset(test_data, transform), batch_size=BATCH_SIZE, num_workers=31)

# Vision Transformer Model
class SimpleViTModule(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = SimpleViT(
            image_size=IMAGE_SIZE,
            patch_size=PATCH_SIZE,
            num_classes=NUM_CLASSES,
            dim=DIM,
            depth=DEPTH,
            heads=HEADS,
            mlp_dim=MLP_DIM
        )
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        acc = (logits.argmax(dim=-1) == y).float().mean()
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=LR)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
        return [optimizer], [scheduler]
    





model = SimpleViTModule()
trainer = pl.Trainer(
    max_epochs=EPOCHS,
    accelerator=DEVICE,
    log_every_n_steps=10,
    check_val_every_n_epoch=1
)



trainer.fit(model, train_loader, test_loader)
