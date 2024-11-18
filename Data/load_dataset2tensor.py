################################################################################################################
# Instruction:

# 1. function "load_dataset2tensor", 

# input :  filename

# output: 
# batch_tensor: batch of images, tensor, (B,3,size 1 of image, size 2 of image)
# labels_tensor: batch of corresponding label, tensor, dimension (B)

################################################################################################################

import torch
import pandas as pd
import io
from PIL import Image
from torchvision import transforms

###################################################################################
def load_dataset2tensor(filename):    
    
    """
    Return batch of images and batch of labels 

    Returns:
        batch_tensor :tensor, (B,3,size 1 of image, size 2 of image)
        labels_tensor: batch of corresponding label, tensor, dimension (B)

    """

    data_file = filename
    df = pd.read_parquet(data_file)

    transform = transforms.ToTensor()

    tensor_list =[]
    labels_list=[]



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
    return batch_tensor, labels_tensor
    


    

############################################################################

if __name__ == "__main__":
    filename="train.parquet"
        # Start the timer
    import time
    start_time = time.time()
    image_tensor, labels_tensor=load_dataset2tensor(filename)
    end_time = time.time()

    # Calculate and print the elapsed time
    elapsed_time = end_time - start_time
    print(f"load dataset and convert to tensor, time taken: {elapsed_time:.6f} seconds")
    print(image_tensor.shape)
    print(labels_tensor.shape)
    
    pass
    

