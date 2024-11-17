
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
    return patch_resize,patch

###############################################################################################################################

if __name__ == "__main__":
    device = torch.device("cpu")


    feature = KF.KeyNetAffNetHardNet(5000, True).eval().to(device)
    img1 = K.io.load_image('../Data/image.jpg', K.io.ImageLoadType.RGB32, device=device)[None, ...]
    with torch.inference_mode():
        lafs1, resps1, descs1 = feature(K.color.rgb_to_grayscale(img1))


    import random
    from scale_angle_rotation import get_laf_scale_and_angle

    # n = 3  # Number of random items you want to pick
    # random_items = random.sample(range(lafs1.shape[1]), n)
    random_items=[80,90,100]

    visualize_LAF(img1, lafs1[:,torch.tensor(random_items),:,:])






    ###############################################################################################################################
    temp_eig,temp_V,temp_angle=get_laf_scale_and_angle(lafs1)   # function ‘get_laf_scale_and_angle‘ should be defined in laf.py
    temp_center=K.feature.laf.get_laf_center(lafs1)   # function ‘get_laf_center‘ is a bulit-in function


    for p in random_items:
        angle = temp_angle[0,p,0].float().item()
        position=temp_center[0,p,:]
        len_major=temp_eig[0,p,0]/2
        len_minor=temp_eig[0,p,1]/2
        size=[16,16]

        temp_resized_patch,temp_patch=get_resized_patch(img1,angle,position,len_major,len_minor,size)
        plt.figure()
        plt.imshow(temp_patch)
        plt.figure()
        plt.imshow(temp_resized_patch)

    plt.pause(0.001)
    input(".")


