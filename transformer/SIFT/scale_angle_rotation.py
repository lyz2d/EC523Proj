##############################################################

# comment by YC: I put this function in laf.py

import cv2
import kornia as K
import kornia.feature as KF
import matplotlib.pyplot as plt
import numpy as np
import torch
from kornia_moons.viz import *
from kornia.core.check import KORNIA_CHECK_LAF, KORNIA_CHECK_SHAPE

def get_laf_scale_and_angle(LAF: torch.Tensor) -> torch.Tensor:
    """Return a scale of the LAFs.

    Args:
        LAF: :math:`(B, N, 2, 3)`

    Returns:
        eig, V, rad2deg(angle_rad).unsqueeze(-1)
        eig :math:`(B, N, 2)`  , eig[B, N, 0] gives you the greatest scale/singular value
        V :math:`(B, N, 2, 12)`, V[:,:,0;1,0:2] or V[:,:,0,0:2] gives you the singular vector correspodning to the greatest scale/singular value
        rad2deg(angle_rad).unsqueeze(-1) :math:`(B, N, 1)`,  the angel between the major axis of oval and the x-axis

        


    """
    KORNIA_CHECK_LAF(LAF)

    centerless_laf = LAF[:, :, :2, :2]
    # B, N = LAF.shape[:2]
    _,eig,V=torch.linalg.svd(centerless_laf[:,:,0:2,0:2])

    # V[:,:,0;1,0:2]  is the sigular vector corresponding to the largest singular value, which is eig[:,:,0]
    temp_1= torch.transpose(V[:,:,0:1,0:2],2,3)
    temp=torch.matmul( centerless_laf[:,:,:,0:2]   ,temp_1   )
   
    angle_rad = torch.atan(temp[..., 1, 0]/temp[..., 0, 0])

    # angle_rad = torch.atan2(temp[..., 1, 0], temp[..., 0, 0])
    # torch.atan2( sin, cos)
    

    return eig, V, np.rad2deg(angle_rad).unsqueeze(-1)





##############################################################################

# K.feature.laf.get_laf_center(lafs1) will give you the center of the oval

###################################################################################

# The following is a simple example to show how to use the output of get_laf_scale_and_angle


if __name__ == "__main__":

    # device = K.utils.get_cuda_or_mps_device_if_available()
    device = torch.device("cpu")


    feature = KF.KeyNetAffNetHardNet(5000, True).eval().to(device)
    img1 = K.io.load_image('../Data/image.jpg', K.io.ImageLoadType.RGB32, device=device)[None, ...]
    with torch.inference_mode():
        lafs1, resps1, descs1 = feature(K.color.rgb_to_grayscale(img1))


    import random
    n = 10  # Number of random items you want to pick
    random_items = random.sample(range(lafs1.shape[1]), n)
    # visualize_LAF(img1, lafs1[:,torch.tensor(random_items),:,:])
    p=10
    visualize_LAF(img1, lafs1[:,[p],:,:])

    i=0

    temp_eig,temp_V,temp_angle=get_laf_scale_and_angle(lafs1)
    # temp1=torch.linspace( 0, temp_eig[0,p,i]/2,10 )
    # temp2=temp_V[0,p,i,0]
    # temp3=temp_V[0,p,i,1]

    temp4=torch.matmul( lafs1[0,p,:,0:2] /2  ,temp_V[0,p,i,0:2]) # get the major axis for oval

    temp5=torch.linspace( 0, temp4[0],10 )
    temp6=torch.linspace( 0, temp4[1],10 )


    # print(temp_angle)

    temp_center=K.feature.laf.get_laf_center(lafs1)   # get the center for oval
    plt.plot(temp5+temp_center[0,p,0],temp6+temp_center[0,p,1] )

    angle_rad = torch.atan(temp4[1]/temp4[0])

    # print(torch.rad2deg(angle_rad))


