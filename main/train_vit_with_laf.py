import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from datasets import load_dataset
from torch.optim import AdamW, Adam, SGD
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from simple_vit_with_laf import ViT
import os


# Hyperparameters
BATCH_SIZE = 64
IMAGE_SIZE = 128  # Assume square images for simplicity
PATCH_SIZE = 16
NUM_CLASSES = 100
DIM = 512
DEPTH = 6
HEADS = 16
MLP_DIM = 1024
DROP_OUT = 0.1
EMB_DROP_OUT = 0.1

MODULE_NAME="ViT_with_laf"
DATASET= "ImageNet_100"
NUM_PATCHES=64

LR = 3e-4
EPOCHS = 150
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CHECK_POINT_EVERY = 25

torch.set_float32_matmul_precision('medium')


laf_train = torch.load('/projectnb/ec523kb/projects/teams_Fall_2024/Team_3/data/lafs_train_IMG128_NUM64.pt')
laf_val = torch.load('/projectnb/ec523kb/projects/teams_Fall_2024/Team_3/data/lafs_val_IMG128_NUM64.pt')


# Data Preparation
train_transforms = transforms.Compose([
    transforms.Lambda(lambda img: img.convert("RGB")),  # Ensure everything converted to 3 channels
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    # transforms.RandomHorizontalFlip(),
    # transforms.RandomCrop(IMAGE_SIZE, padding=4),
    transforms.ToTensor(),
    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
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
using_scc  = True
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
        return image, label,idx

# DataLoader
train_loader = DataLoader(CustomImageDataset(train_data, train_transforms), batch_size=BATCH_SIZE, shuffle=True, num_workers=31)
test_loader = DataLoader(CustomImageDataset(test_data, val_transforms), batch_size=BATCH_SIZE, num_workers=31)






# Vision Transformer Model
class SimpleViTModule(pl.LightningModule):
    def __init__(self):
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
            "epochs": EPOCHS,
            "number of patches": NUM_PATCHES,
            "module name": MODULE_NAME,
            "dataset": DATASET
        })
        self.model = ViT(
            image_size = IMAGE_SIZE,
            patch_size = PATCH_SIZE,
            num_classes = NUM_CLASSES,
            dim = DIM,
            depth = DEPTH,
            heads = HEADS,
            mlp_dim = MLP_DIM,
            dropout = DROP_OUT,
            emb_dropout = EMB_DROP_OUT,
            num_patches=NUM_PATCHES
        )
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x,laf):
        return self.model(x,laf)

    def training_step(self, batch, batch_idx):
        x, y ,idx = batch
        laf=torch.zeros(x.shape[0],NUM_PATCHES,2,3).to(DEVICE)

        for i in range(x.shape[0]):
            laf[i,:,:,:]=laf_train[idx[i]]
 
        logits = self(x,laf)
        loss = self.criterion(logits, y)
        self.log("train_loss", loss, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y,idx = batch

        laf=torch.zeros(x.shape[0],NUM_PATCHES,2,3).to(DEVICE)

        for i in range(x.shape[0]):
            laf[i,:,:,:]=laf_val[idx[i]]


        logits = self(x,laf)
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
    



model = SimpleViTModule()
#model = SimpleViTModule.load_from_checkpoint('/home/yge/deep_learning/EC523Proj/lightning_logs/version_6/checkpoints/epoch=49-step=13050.ckpt')
model = SimpleViTModule()

logger = TensorBoardLogger(save_dir='./imagenet100_log', name='ViT_with_laf_logs')
checkpoint_callback = ModelCheckpoint(
    save_top_k=-1,  # Save all checkpoints
    every_n_epochs=CHECK_POINT_EVERY,  # Save every 25 epochs
    dirpath='./imagenet100_log/checkpoints',  # Path to save checkpoints
    filename='epoch-{epoch:02d}-val_loss-{val_loss:.4f}',  # Filename format
    save_weights_only=False  # Save the entire model
)

trainer = pl.Trainer(
    max_epochs=EPOCHS,
    accelerator=DEVICE,
    log_every_n_steps=10,
    check_val_every_n_epoch=1,
    devices = 'auto',
    logger=logger,  # Add the logger here
    callbacks=[checkpoint_callback]  # Add the checkpoint callback here
    )



trainer.fit(model, train_loader, test_loader)