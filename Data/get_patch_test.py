
import torchvision

import cv2
import kornia as K
import kornia.feature as KF
import matplotlib.pyplot as plt
import numpy as np
import torch
from kornia_moons.viz import *

#############################################################################################################
def get_resized_patch(img,angle,position,len_major,len_minor,size=[16,16]):
    
    """Return a resize(distorted) patch of given size.

    Args:
        img: :`(1, 3, size 1 of img, size 2 of img)`
        angle : 'float' (not tensor), in degree instead of rad
        position: tensor (1 dimensional) of length 2, center of the oval/unresized patch
        len_major: the length of the major axis of the oval
        len_minor: the length of the minor axis of the oval
        size: list of length 2, the size of the desired resized patch, e.g.[16,16]
        

    Returns:
        resized_patch: [size[0],size[1],3]

        
    """
    rotated_img=torchvision.transforms.functional.rotate(img,angle)

    # print(rotated_img.shape)
    # print(img1.shape)
    # plt.imshow(rotated_img)
    # visualize_LAF(rotated_img, lafs1[:,[p],:,:])
    # Rotation angle in degrees
    # angle = torch.rad2deg(angle_rad).float().item()

    # Image center (e.g., center of rotation)
    cx, cy = img.shape[2]/2, img.shape[3]/2

    # Get the rotation matrix
    rotation_matrix = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
    rotation_matrix=torch.from_numpy(rotation_matrix)
    rotation_matrix=rotation_matrix.to(torch.float)

    # adding 1 dimension renders [x,y] --> [x,y,1]
    # original_point=torch.cat([temp_center[0,p,:],torch.ones_like(temp_center[0,p,0:1])])
    original_point=torch.cat([position,torch.ones_like(position[0:1])])
    original_point=original_point.unsqueeze(-1)

    # Calculate the new position by multiplying the rotation matrix
    new_point = torch.matmul(rotation_matrix, original_point)
    new_x, new_y = new_point[0,:], new_point[1,:]


    patch=rotated_img[:,:,
                    torch.max(torch.cat([new_y-len_minor,torch.tensor([0]) ]) ) .ceil().to(torch.int): 
                    torch.min( torch.cat([new_y+len_minor,torch.tensor( rotated_img.shape[3:4]) ]) ).floor().to(torch.int),
                    torch.max(torch.cat([new_x-len_major,torch.tensor([0]) ]) ).ceil().to(torch.int): 
                    torch.min( torch.cat([new_x+len_major,torch.tensor( rotated_img.shape[2:3]) ]) ).floor().to(torch.int) 
                    ]    # the first two dimensions of patch match the single input img, which is [1,3,:,:]



    patch=patch.permute(2,3,1,0)
    patch=patch.squeeze(-1)  # the dimension of patch is [:,:,3]


    patch_resize=torchvision.transforms.functional.resize(patch.permute(2,0,1), size)
    patch_resize=patch_resize.permute(1,2,0) # the dimension of patch is [70,70,3]
    # plt.imshow(p)
    return patch_resize

###############################################################################################################################
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
#########################################################################################



device = torch.device("cpu")


feature = KF.KeyNetAffNetHardNet(5000, True).eval().to(device)



import time

# Start the timer
start_time = time.time()



LAFs=torch.load('1k_LAFs_from_train_set.pt')   #list of tensor, lafs for the first 1000 images in the training set 
# print(len(LAFs))
# print(LAFs[5].shape)

tokens=[]
for i in range(len(LAFs)):
    lafs=LAFs[i]
    img=batch_tensor[i:i+1,:,:,:]
    temp_eig,temp_V,temp_angle=K.feature.laf.get_laf_scale_and_angle(lafs)   # function ‘get_laf_scale_and_angle‘ should be defined in laf.py
    temp_center=K.feature.laf.get_laf_center(lafs)   # function ‘get_laf_center‘ is a bulit-in function
    # use a to control the size of original patch
    a=0.5
        
    token_from_img=[]

    for p in range(lafs.shape[1]):
        angle = temp_angle[0,p,0].float().item()
        position=temp_center[0,p,:]
        len_major=temp_eig[0,p,0]*a
        len_minor=temp_eig[0,p,1]*a
        size=[16,16]

        temp_resized_patch=get_resized_patch(img,angle,position,len_major,len_minor,size)
        token_from_img.append(temp_resized_patch)
    
    tokens.append(token_from_img)
 
# tokens[i] contains a list of patch from the i-th image, tokens[i][j] is the patch corresponding to the j-th key points of i-th image        
        
        

end_time = time.time()

# Calculate and print the elapsed time
elapsed_time = end_time - start_time
print(f"process {len(LAFs):.6f} images in total, time taken: {elapsed_time:.6f} seconds")
    








    