##########################################################################################################
# Important: please read this part and ignore the previous instructions, if we decide to use 
# Pytorch built-in function "grid_sample" to get patches

# Now, we only need to call 3 functions: (1) to compute LAF during the training loop; (2) to get the patches during the training loop;
# (3) to get the features (scale, angle, position) for each patch, based on the LAF. 

# 1.  function "get_lafs_for_batch_images":
# input :
# (1)batch_images, tensor of size (B,3,H,W)
# (2)max_point_num, integer, the default value is 64

# Output:
# batch_lafs: tensor of shape '(B,max_point_num,2,3)'


# 2. function "get_patches_for_batch_images"
# input:
# (1)batch_images, tensor of size  '(B,3,H,W)'
# (2)LAFs_tensor, tensor of shape '(B,max_point_num,2,3)'
# (3)size_resize, the size of the desired patch, default: [16,16]
# (4)max_point_num, integer, the default value is 64

# output: 
# patch_tensor, tensor, size: '(B, max_point_num, 3, size_resize[0], size_resize[1])'
    
# example: [b,p,:,:,:] is the resized patch of the p-th key point of the b-th image in the batch; 
# if the b-th image only has P key points, where P< maximum number of keypoint, then [b,P+1,:,:,:] is tensor of zeros

# 3. function "get_feature_from_LAF", 
# input: LAFs for batch of images,  tensor of size '(B, max_point_num, 2, 3)'

# output: (1)scale (the size of the ellipsoid), (2)angle, (3) center (position in image)
# B: batch, max_point_num: maximum number of key points
# scale: tensor:`(B, max_point_num, 2)`  , eig[B, N, 0] gives you the greatest scale/singular value
# angle: tensor:`(B, max_point_num, 1)`,  the angel between the major axis of oval and the x-axis
# center: tensor:`(B, max_point_num, 2)`
    
    




################################################################################################################
# Instructions:

# 1. function "get_resized_patch"
# input: 
# (1) img: :tensor `(1, 3, size 1 of img, size 2 of img)`
# (2) angle : 'float' (not tensor), in degree instead of rad
# (3) position: tensor (1 dimensional) of length 2, center of the oval/unresized patch
# (4) len_major: torch, 0 dimension, the length of the major axis of the oval
# (5) len_minor: torch, 0 dimension, the length of the minor axis of the oval
# (6) size: list of length 2, the required size of the resized patch, default: [16,16]
        
# output: corresponding resized patch (tensor: (size[0], size[1], 3))

# 2. function "get_patch_for_dataset", 
# This function goes over the images of the dataset and creates a list of tokens, where each row has the patches (tokens) of each image. The row
# length varies per image. Each position in the row (tokens[i][j]) by a patch (token) represented as a tensor.
# input : 
# (1) batch of images(tensor) 
# (2) corresponding LAFs(tensor);
# (3) size_original: determine how large is the region used to generate patch, default:0.5
# (4) size_resize: the size of the desired resized patch, default: [16,16]
         
# output: tokens (patches),         
# token: a list of lists of tokens for the data set
# tokens[i] contains a list of all patches from the (i+1)-th image.
# tokens[i][j]: the patch corresponding to the (j+1)-th key points of (i+1)-th image. tensor: [size_resize[0],size_resize[1],3] for any i,j

# 3. function "get_laf_scale_and_angle", input: LAFs for batch of images; output: (1)eig, (2)V, (3)angle, where
# eig :math:`(B, N, 2)`  , eig[B, N, 0] gives you the greatest scale/singular value
# V :math:`(B, N, 2, 2)`, V[:,:,0;1,0:2] or V[:,:,0,0:2] gives you the singular vector correspodning to the greatest scale/singular value
# angle :math:`(B, N, 1)`,  the angel between the major axis of oval and the x-axis


# 4. function "get_feature_from_LAF", 
# input: LAFs for batch of images; 
# output: (1)scale (the size of the ellipsoid), (2)angle, (3) center (position in image)
# B: batch, N: number of key points
# scale: tensor:`(B, N, 2)`  , eig[B, N, 0] gives you the greatest scale/singular value
# angle: tensor:`(B, N, 1)`,  the angel between the major axis of oval and the x-axis
# center: tensor:`(B, N, 2)`





# Please note that V is not the axis of oval




################################################################################################################




import torchvision
import cv2
import torch

########################################################################

import kornia.feature as KF
import kornia as K
import numpy as np

def get_lafs_for_batch_images(batch_images,max_point_num=64):
    """""
    compute lafs directly from a batch of images
    
    Args:
    batch_images: tensor of size: (B,3,H,W)
    
    return:
    batch_lafs: tensor of shape (B,max_point_num,2,3)
    """
    
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    device= batch_images.device
    
    keynet_feature = KF.KeyNetAffNetHardNet(5000, True).eval().to(device)

    batch, _, _, _ = batch_images.shape

    batch_lafs=torch.zeros(batch, max_point_num, 2, 3).to(device)
        

    for i in range(batch):
        img=batch_images[i:i+1,:,:,:]
        with torch.inference_mode():
            lafs_keynet, resps_keynet, descs_keynet = keynet_feature(K.color.rgb_to_grayscale(img))
            
        lafs_keynet=lafs_keynet.squeeze(0)
        if lafs_keynet.shape[0]>max_point_num:
            items=np.linspace(0,lafs_keynet.shape[0]-1,max_point_num, dtype=int)
            batch_lafs[i,:,:,:]=lafs_keynet[torch.tensor(items),:,:]
        else:
            batch_lafs[i,0:lafs_keynet.shape[0],:,:]=lafs_keynet
    return batch_lafs
        


import torch.nn.functional as F

def get_patches_for_batch_images(batch_images,LAFs_tensor,size_resize=[16,16], max_point_num=64):
    
    """

    Args:
        batch_images: :`(B, 3, size 1 of img, size 2 of img)`
        LAFs:  tensor of size (B,max_point_num,2,3 )
        size_resize:list of length 2, the size of the desired resized patch, default: [16,16]
        max_point_num: integer, maximum number of key point
        
        

    Returns: patch_tensor, tensor, size: '(B, maximum number of keypoint, 3, size_resize[0], size_resize[1])'
    
    example: [b,p,:,:,:] is the resized patch of the p-th key point of the b-th image in the batch; 
    if the b-th image only has P key points, where P< maximum number of keypoint, then [b,P+1,:,:,:] is tensor of zeros

    """
    device= batch_images.device
    batch, _, _, _ = batch_images.shape
    assert batch==LAFs_tensor.shape[0],  "batch size of batch_images doesn't match batch size of LAFs"
    assert max_point_num==LAFs_tensor.shape[1], "please make sure the max_num_point are consistent throughout the process"
    

    # batch_images must be of size (B,3,H ,W)
    # LAFs_tensor must be of size (B,max_point_num,2,3 )
    
    patch_tensor=torch.zeros(batch, max_point_num,3, size_resize[0], size_resize[1]).to(device)
    
    temp1=torch.tensor([batch_images.shape[-2],batch_images.shape[-1]]).to(device)
    temp1=temp1/2
    temp1=temp1.view(1,1,1,2)

    
    for i in range(max_point_num):
        grid = F.affine_grid(LAFs_tensor[:,i,:,:],torch.Size((batch, 3, size_resize[0], size_resize[1])),align_corners = False).to(device)
        grid_normalized=grid/temp1
        grid_normalized=grid_normalized-torch.tensor([1,1]).view(1,1,1,2)
        patch_tensor[:,i,:,:,:] = F.grid_sample(input=batch_images, grid=grid_normalized,mode='bilinear')
        
    return patch_tensor


###########################################################################################################
def list_lafs_to_tensor(LAFs,max_point_num=64):
    
    """
    Recall that we saved lafs for each images as a list of tensor, each elment (i.e. LAFs[i]) is a tensor of shape (1, N,2,3),
    this function will return a lafs_tensor of shape (B,max_point_num,2,3 ) through zeros-padding
    """
    
    assert type(LAFs)==list
    
    LAFs_tensor=torch.zeros(len(LAFs),max_point_num,2,3)
    # torch.nn.utils.rnn.pad_sequence(LAFs,batch_first=True)
    
    
    for i in range(len(LAFs)):
        if LAFs[i].shape[1]>max_point_num:
            print(f"the number of key points for some images is greater than the given maximum number of key point.")
            
            # to do: how to truncate if there are too many key points
            
        else:
            LAFs_tensor[i,0:LAFs[i].shape[1],:,:]=LAFs[i].squeeze(0)
        
    return LAFs_tensor
    

############################################################

def get_rotation_matrix_2d(center, angle, scale):
    """
    PyTorch getRotationMatrix2D
    Parameters:
        center: (float, float) 
        angle: float 
        scale: float 
    Return:
        2x3 rotation matrix
    """
    # Rad to degree
    angle_rad = torch.deg2rad(torch.tensor(angle))
    
    # Rotation matrix elements
    alpha = scale * torch.cos(angle_rad)
    beta = scale * torch.sin(angle_rad)
    
    # Rotation matrix
    matrix = torch.tensor([
        [alpha, beta, (1 - alpha) * center[0] - beta * center[1]],
        [-beta, alpha, beta * center[0] + (1 - alpha) * center[1]]
    ])
    
    return matrix



#############################################################################################################
def get_resized_patch(img,angle,position,len_major,len_minor,size=[16,16],device = 'cuda'):
    
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

    # Image center (e.g., center of rotation)
    cx, cy = img.shape[2]/2, img.shape[3]/2

    # Get the rotation matrix
    rotation_matrix = get_rotation_matrix_2d((cx, cy), angle, 1.0).to(device)
    #rotation_matrix=torch.from_numpy(rotation_matrix)
    #rotation_matrix=rotation_matrix.to(torch.float)

    # adding 1 dimension renders [x,y] --> [x,y,1]
    # original_point=torch.cat([temp_center[0,p,:],torch.ones_like(temp_center[0,p,0:1])])
    original_point=torch.cat([position,torch.ones_like(position[0:1])])
    original_point=original_point.unsqueeze(-1)

    # Calculate the new position by multiplying the rotation matrix
    new_point = torch.matmul(rotation_matrix, original_point)
    new_x, new_y = new_point[0,:], new_point[1,:]
    
    patch=rotated_img[:,:,
                    torch.max(torch.cat([new_y-len_minor,torch.tensor([0]).cuda() ]) ) .ceil().to(torch.int): 
                    torch.min( torch.cat([new_y+len_minor,torch.tensor( rotated_img.shape[3:4]).cuda() ]) ).floor().to(torch.int),
                    torch.max(torch.cat([new_x-len_major,torch.tensor([0]).cuda() ]) ).ceil().to(torch.int): 
                    torch.min( torch.cat([new_x+len_major,torch.tensor( rotated_img.shape[2:3]).cuda() ]) ).floor().to(torch.int) 
                    ]    # the first two dimensions of patch match the single input img, which is [1,3,:,:]



    patch=patch.permute(2,3,1,0)
    patch=patch.squeeze(-1)  # the dimension of patch is [:,:,3]


    patch_resize=torchvision.transforms.functional.resize(patch.permute(2,0,1), size)
    patch_resize=patch_resize.permute(1,2,0) # the dimension of patch is [70,70,3]
    # plt.imshow(p)
    return patch_resize

###############################################################################################################################
def get_patch_for_dataset(batch_images,LAFs,size_original=0.5,size_resize=[16,16], max_point_num=64):
    
    """
    Return the tokens for a batch of images, where one image is of size (3, size 1 of img, size 2 of img)

    Args:
        batch_images: :`(B, 3, size 1 of img, size 2 of img)`
        LAFs: list of lafs tensor, LAFs[i] is the lafs tensor of dimension (1,number of key point,2,3) for (i+1)-th images
        size_original: determine how large is the region used to generate patch
        size_resize:list of length 2, the size of the desired resized patch, default: [16,16]
        max_point_num: integer, maximum number of key point
        
        

    Returns: patch_tensor, tensor, size: '(B, maximum number of keypoint, size_resize[0], size_resize[1], 3)'
    example: [b,p,:,:,:] is the resized patch of the p-th key point of the b-th image in the batch; 
    if the b-th image only has P key points, where P< maximum number of keypoint, then [b,P+1,:,:,:] is tensor of zeros

    """
    batch, _, _, _ = batch_images.shape
    assert batch==len(LAFs)
    
    patch_tensor=torch.zeros(batch, max_point_num, size_resize[0], size_resize[1], 3).cuda()
    
    tokens=[]
    for i in range(len(LAFs)):
        lafs=LAFs[i]
        img=batch_images[i:i+1,:,:,:]
        temp_eig,temp_V,temp_angle=get_laf_scale_and_angle(lafs)  
        temp_center=get_laf_center(lafs)   
        # use size_patch to control the size of original patch
            
        #token_from_img=[]

        for p in range(lafs.shape[1]):
            angle = temp_angle[0,p,0].float().item()
            position=temp_center[0,p,:]
            len_major=temp_eig[0,p,0]*size_original
            len_minor=temp_eig[0,p,1]*size_original
            size=size_resize

            temp_resized_patch=get_resized_patch(img,angle,position,len_major,len_minor,size)
            # token_from_img.append(temp_resized_patch)
            if p < max_point_num:
                patch_tensor[i,p,:,:,:]=temp_resized_patch
        
        #tokens.append(token_from_img)
    return patch_tensor
 
################################################################################################################


def get_laf_scale_and_angle(LAF):
    """Return  scale and angle of the LAFs.

    Args:
        LAF: :tensor:`(B, N, 2, 3)`

    Returns:
        eig :tensor:`(B, N, 2)`  , eig[B, N, 0] gives you the greatest scale/singular value
        V :tensor:`(B, N, 2, 2)`, V[:,:,0;1,0:2] or V[:,:,0,0:2] gives you the singular vector correspodning to the greatest scale/singular value
        angle_deg :tensor:`(B, N, 1)`,  the angel between the major axis of oval and the x-axis


    """
    device=LAF.device

    centerless_laf = LAF[:, :, :2, :2]
    _,eig,V=torch.linalg.svd(centerless_laf[:,:,0:2,0:2])

    # V[:,:,0;1,0:2]  is the sigular vector corresponding to the largest singular value, which is eig[:,:,0]
    temp_1= torch.transpose(V[:,:,0:1,0:2],2,3)
    temp=torch.matmul( centerless_laf[:,:,:,0:2]   ,temp_1   )

    angle_rad = torch.atan(temp[..., 1, 0]/temp[..., 0, 0]).to(device)
    pi = torch.tensor(3.14159265358979323846).to(device)
    angle_deg=180.0 * angle_rad / pi.type(angle_rad.dtype)
    angle_deg=angle_deg.unsqueeze(-1)

    return eig, V, angle_deg

###################################################################################################################

def get_laf_center(LAF):
    """Return a center (keypoint) of the LAFs. The convention is that center of 5-pixel image (coordinates from 0
    to 4) is 2, and not 2.5.

    Args:
        LAF: :tensor:`(B, N, 2, 3)`

    Returns:
        output :tensor:`(B, N, 2)`

    Example:
        >>> input = torch.ones(1, 5, 2, 3)  # BxNx2x3
        >>> output = get_laf_center(input)  # BxNx2
    """

    out = LAF[..., 2]
    return out

#########################################################################################################################

def get_feature_from_LAF(LAF):
    """Return a  scale,angle, center (keypoint) of the LAFs. 
    Args:
        LAF: :tensor:`(B, N, 2, 3)`

    Returns:
        scale :tensor:`(B, N, 2)`  , eig[B, N, 0] gives you the greatest scale/singular value
        angle:tensor:`(B, N, 1)`,  the angel between the major axis of oval and the x-axis
        center :tensor:`(B, N, 2)`

    Example:
        >>> input = torch.ones(1, 5, 2, 3)  # BxNx2x3
        >>> scale,angle,center = get_feature_from_LAF(input)  # 
    """

    scale, V, angle=get_laf_scale_and_angle(LAF)
    center=get_laf_center(LAF)
    
    return scale,angle,center

##########################################################################################

if __name__ == "__main__":
    
    import load_dataset2tensor as load_dataset
    
    filename="../Data/train.parquet"
    # Start the timer
    import time
    start_time = time.time()
    image_tensor, labels_tensor=load_dataset.load_dataset2tensor(filename)
    end_time = time.time()

    # Calculate and print the elapsed time
    elapsed_time = end_time - start_time
    print(f"load dataset and convert to tensor, time taken: {elapsed_time:.6f} seconds")
    print(image_tensor.shape)
    print(labels_tensor.shape)



    device = torch.device("cpu")

    import time

    # Start the timer
    start_time = time.time()

    LAFs = torch.load('../Data/LAFs_from_train_set.pt', map_location=torch.device('cpu'))
    LAFs=LAFs[0:100]
    

    print(len(LAFs))

    tokens=get_patch_for_dataset(image_tensor[0:100],LAFs)
    print(tokens.shape)
    # tokens[i] contains a list of patch from the i-th image, tokens[i][j] is the patch corresponding to the j-th key points of i-th image        
            
    end_time = time.time()

    # Calculate and print the elapsed time
    elapsed_time = end_time - start_time

    # torch.save(tokens, 'patches_from_train_set.pt') 
    print(f"process {len(LAFs):.6f} images in total, time taken: {elapsed_time:.6f} seconds")
        

    pass












    
