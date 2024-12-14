import torch
from datasets import load_dataset
from tqdm import tqdm
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, TensorDataset
from torchvision.datasets import ImageFolder
from SIFT.get_patch_and_feature import get_lafs_for_batch_images
import kornia.feature as KF
import time

IMG_SIZE = 128
BATCH_SIZE = 64
MAX_POINT_NUM = 128

dataset = load_dataset("ilee0022/ImageNet100", cache_dir='/projectnb/ec523kb/projects/teams_Fall_2024/Team_3/hg_cache')
train_img = dataset['train']
test_img = dataset['validation']

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
        return image, label, idx

def add_laf(example, laf):
    example['laf'] = (laf)
    return example

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.Lambda(lambda img: img.convert("RGB")),  # Ensure everything converted to 3 channels
    transforms.ToTensor(),
    # transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])
# DataLoader
train_loader = DataLoader(CustomImageDataset(train_img, transform), batch_size=BATCH_SIZE, shuffle=False, num_workers=31)
val_loader = DataLoader(CustomImageDataset(test_img, transform), batch_size=BATCH_SIZE, num_workers=31)
print(len(train_loader))

# Iterate through the DataLoader and compute LAF
lafs_list_train = [None] * len(train_img)  # Pre-allocate a list to store LAFs based on dataset size
lafs_list_val = [None] * len(test_img) 

KFLAF = KF.ScaleSpaceDetector(
    num_features=500,      # Number of keypoints to detect
    # scale_pyramid_levels=3,  # Levels of the scale-space pyramid
    # init_scale=1.6,        # Initial scale of the pyramid
    # response_threshold=0.01  # Minimum response threshold
)

# Detect keypoints

for batch_idx, (images, labels, indices) in enumerate(tqdm(train_loader, desc="Processing Batches")):
    lafs = KFLAF(images)  # Compute LAF for batch
    for idx, laf in zip(indices, lafs):
        lafs_list_train[idx] = laf  # Store each LAF in the list using the dataset index
torch.save(lafs_list_train, "lafs_train.pt")

for batch_idx, (images, labels, indices) in enumerate(tqdm(val_loader, desc="Processing Batches")):
     lafs = get_lafs_for_batch_images(images, max_point_num=MAX_POINT_NUM)  # Compute LAF for batch
     for idx, laf in zip(indices, lafs):
         lafs_list_val[idx] = laf.cpu()  # Store each LAF in the list using the dataset index
torch.save(lafs_list_val, "lafs_val.pt")





