import pandas as pd
import io
from PIL import Image
import torch
from torchvision import transforms
import kornia as K
import matplotlib.pyplot as plt


import torchvision

import cv2
import kornia as K
import kornia.feature as KF
import matplotlib.pyplot as plt
import numpy as np
import torch
from kornia_moons.viz import *

# Step 1: Load the parquet file
data_file = "train.parquet"
df = pd.read_parquet(data_file)

transform = transforms.ToTensor()

tensor_list =[]
labels_list=[]



# Start the timer
import time
start_time = time.time()

for index, row in df['image'].items():
    img_bytes = row['bytes']  # Access the bytes from the dictionary
    img = Image.open(io.BytesIO(img_bytes))  # Convert bytes to a PIL Image
    
    # Now you can process `img`, such as converting it to a tensor
    img_tensor = transform(img)
    if img_tensor.shape[0]==3:
        tensor_list.append(img_tensor)
        labels_list.append(df.loc[index, 'label'])

labels_tensor=torch.tensor(labels_list)
batch_tensor = torch.stack(tensor_list)


end_time = time.time()

# Calculate and print the elapsed time
elapsed_time = end_time - start_time
print(f"Time taken: {elapsed_time:.6f} seconds")


print(batch_tensor.shape)
print(labels_tensor.shape)


###################################################################################################################
# Using the following command to save and load, but the size of file will be large (4.5G)

# torch.save(batch_tensor, 'train_images.pt') 
# torch.save(labels_tensor, 'train_labels.pt') 
# temp1= torch.load('train_images.pt')
# temp2= torch.load('train_labels.pt')


##################################################################################################################






