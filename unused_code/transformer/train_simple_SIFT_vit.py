import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from datasets import load_dataset
from torch.optim import AdamW, Adam, SGD
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from simple_vit import ViT, SIFT_ViT, Ssd_ViT
import os
from kornia.feature import harris_response

torch.autograd.set_detect_anomaly(True)



# Hyperparameters
BATCH_SIZE = 128
IMAGE_SIZE = 256  # Assume square images for simplicity
PATCH_SIZE = 16
NUM_CLASSES = 100
DIM = 512
DEPTH = 6
HEADS = 16
MLP_DIM = 1024
DROP_OUT = 0.1
EMB_DROP_OUT = 0.1

LR = 3e-4
EPOCHS = 75
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CHECK_POINT_EVERY = 25

torch.set_float32_matmul_precision('medium')

# Data Preparation
train_transforms = transforms.Compose([
    transforms.Lambda(lambda img: img.convert("RGB")),  # Ensure everything converted to 3 channels
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(IMAGE_SIZE, padding=4),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transforms = transforms.Compose(
    [
        transforms.Lambda(lambda img: img.convert("RGB")),
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
    ]
)

dataset_name = "ilee0022/ImageNet100"

# Load dataset and split
using_scc  = False
#If using scc to train it, 
if using_scc:
    dataset = load_dataset(dataset_name, cache_dir='/projectnb/ec523kb/projects/teams_Fall_2024/Team_3/hg_cache')
else:
    dataset = load_dataset(dataset_name)

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
train_loader = DataLoader(CustomImageDataset(train_data, train_transforms), batch_size=BATCH_SIZE, shuffle=True, num_workers=31)
test_loader = DataLoader(CustomImageDataset(test_data, val_transforms), batch_size=BATCH_SIZE, num_workers=31)

# Vision Transformer Model
class SimpleViTModule(pl.LightningModule):
    def __init__(self, detector):
        super().__init__()
        self.save_hyperparameters({
            "batch_size": BATCH_SIZE,
            "image_size": IMAGE_SIZE,
            "patch_size": PATCH_SIZE,
            "num_classes": NUM_CLASSES,
            "dim": DIM,
            "depth": DEPTH,
            "heads": HEADS,
            "mlp_dim": MLP_DIM,
            "DROP_OUT" : DROP_OUT,
            "EMB_DROP_OUT" : EMB_DROP_OUT,
            "learning_rate": LR,
            "epochs": EPOCHS
        })
        self.model = Ssd_ViT(
            detector=detector,
            image_size = IMAGE_SIZE,
            patch_size = PATCH_SIZE,
            num_classes = NUM_CLASSES,
            dim = DIM,
            depth = DEPTH,
            heads = HEADS,
            mlp_dim = MLP_DIM,
            dropout = DROP_OUT,
            emb_dropout = EMB_DROP_OUT
        )
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        self.log("train_loss", loss, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        acc = (logits.argmax(dim=-1) == y).float().mean()
        self.log("val_loss", loss, prog_bar=True, sync_dist=True)
        self.log("val_acc", acc, prog_bar=True, sync_dist=True)
        return loss

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=LR)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
        return [optimizer], [scheduler]
        #return optimizer
    



#model = SimpleViTModule.load_from_checkpoint('/home/yge/deep_learning/EC523Proj/lightning_logs/version_6/checkpoints/epoch=49-step=13050.ckpt')
model = SimpleViTModule(harris_response)

logger = TensorBoardLogger(save_dir='./imagenet100_log', name='Ssd_ViT_logs')
checkpoint_callback = ModelCheckpoint(
    save_top_k=-1,  # Save all checkpoints
    every_n_epochs=CHECK_POINT_EVERY,  # Save every 25 epochs
    dirpath='./imagenet100_log/Ssd_ViT_checkpoints',  # Path to save checkpoints
    filename='epoch-{epoch:02d}-val_loss-{val_loss:.4f}',  # Filename format
    save_weights_only=False  # Save the entire model
)

trainer = pl.Trainer(
    max_epochs=EPOCHS,
    accelerator=DEVICE,
    log_every_n_steps=10,
    check_val_every_n_epoch=1,
    devices = 'auto',
    strategy='ddp_find_unused_parameters_true',
    logger=logger,  # Add the logger here
    callbacks=[checkpoint_callback]  # Add the checkpoint callback here
    )



trainer.fit(model, train_loader, test_loader)
