# EC523Proj

This repository is for the BU EC523 Project, where implement a Scale-invariant feature transform (SIFT)-based Transformer for image classification. Here we guide tokenization/patching with SIFT as the local feature detection algorithm. These patches are featurized according to their size, orientation, and position. The main part is included in the file, transformer, which has the SIFT code, Transformer code, and the related data set, Tiny_ImageNet. The following figure is a process illustration.
![The SIFT-based Transformer framework running process](./process.png)

# How to Use
Run the `train.py` file to train the model. This will automatically make the image patches and features. 
## Content
### SIFT
First get patches from all images in the dataset using the `get_patch_and_feature.py` script. This will output a list of tokens, where each row *i* is an image with variable number of patches. Each patch (token[i][j]) is represented as a tensor.
### Transformer

### Dataset
The dataset used is TinyImagenet, which has 110k 64x64 colored images with 200 classes. 
