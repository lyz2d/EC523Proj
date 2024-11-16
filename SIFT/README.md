An example about how to load the training dataset with .parquet file using Kornia. 
Please try to adjust it and fit it into get_patch.py and scale_angle_rotation.py


import pandas as pd
import base64
import io
from PIL import Image
import torch
import torchvision.transforms as T
import kornia as K

# Step 1: Load the parquet file
data_file = "path_to_your/train.parquet"
df = pd.read_parquet(data_file)

# Assuming the dataframe has a column 'image' containing base64 strings
images_base64 = df['image']

# Step 2: Decode base64 images and convert to PyTorch tensors
def decode_base64_to_tensor(base64_str):
    # Decode the base64 string
    image_bytes = base64.b64decode(base64_str)
    # Open the image
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    # Convert to tensor
    transform = T.ToTensor()  # Normalize to [0, 1] range
    return transform(image)

# Step 3: Apply the decode function to all rows
image_tensors = [decode_base64_to_tensor(img) for img in images_base64]

# Step 4: Stack into a single tensor for batch processing
images_batch = torch.stack(image_tensors)

# Step 5: Use Kornia for additional preprocessing (e.g., resizing, normalization)
# Resize to 224x224 and normalize with ImageNet stats
transform = K.augmentation.Resize((224, 224))
images_resized = transform(images_batch)

mean = torch.tensor([0.485, 0.456, 0.406])  # ImageNet mean
std = torch.tensor([0.229, 0.224, 0.225])   # ImageNet std

images_normalized = K.enhance.normalize(images_resized, mean, std)

# images_normalized is now ready for use in training!
