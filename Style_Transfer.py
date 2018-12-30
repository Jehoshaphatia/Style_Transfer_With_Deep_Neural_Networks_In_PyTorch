#!/usr/bin/env python
# coding: utf-8

# # Style Transfer with Deep Neural Networks In PyTorch
# 
# 
# In this project, I recreated a style transfer method that is outlined in the paper, [Image Style Transfer Using Convolutional Neural Networks, by Gatys](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Gatys_Image_Style_Transfer_CVPR_2016_paper.pdf) in PyTorch.
# 
# In the paper, style transfer uses the features found in the 19-layer VGG Network, which is comprised of a series of convolutional and pooling layers, and a few fully-connected layers. Conv_1_1 is the first convolutional layer that an image is passed through, in the first stack. Conv_2_1 is the first convolutional layer in the *second* stack. The deepest convolutional layer in the network is conv_5_4.

# ### Separating Style and Content
# 
# Style transfer relies on separating the content and style of an image. Given one content image and one style image, my aim is to create a new, _target_ image which will contain my desired content and style components:
# * objects and their arrangement are similar to that of the **content image**
# * style, colors, and textures are similar to that of the **style image**
# In this project, I'll use a pre-trained VGG19 Net to extract content or style features from a passed in image. I'll then formalize the idea of content and style _losses_ and use those to iteratively update my target image until I get a result that I want.

# import resources
get_ipython().run_line_magic('matplotlib', 'inline')

from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.optim as optim
from torchvision import transforms, models


# ## Load in VGG19 (features)
# 
# VGG19 is split into two portions:
# * `vgg19.features`, which are all the convolutional and pooling layers
# * `vgg19.classifier`, which are the three linear, classifier layers at the end
# 
# I only need the `features` portion, which i'm going to load in and "freeze" the weights of, below.

# get the "features" portion of VGG19 (I will not need the "classifier" portion)
vgg = models.vgg19(pretrained=True).features

# freeze all VGG parameters since i'm only optimizing the target image
for param in vgg.parameters():
    param.requires_grad_(False)


# move the model to GPU, if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

vgg.to(device)


# ### Load in Content and Style Images
# 
# With the helper function below I can load in any type and size of image. The `load_image` function also converts images to normalized Tensors.
# 
# Additionally, it will be easier to have smaller images and to squish the content and style images so that they are of the same size.

def load_image(img_path, max_size=400, shape=None):
    ''' Load in and transform an image, making sure the image
       is <= 400 pixels in the x-y dims.'''
    
    image = Image.open(img_path).convert('RGB')
    
    # large images will slow down processing
    if max(image.size) > max_size:
        size = max_size
    else:
        size = max(image.size)
    
    if shape is not None:
        size = shape
        
    in_transform = transforms.Compose([
                        transforms.Resize(size),
                        transforms.ToTensor(),
                        transforms.Normalize((0.485, 0.456, 0.406), 
                                             (0.229, 0.224, 0.225))])

    # discard the transparent, alpha channel (that's the :3) and add the batch dimension
    image = in_transform(image)[:3,:,:].unsqueeze(0)
    
    return image


# Next, I'm loading in images by file name and forcing the style image to be the same size as the content image.

# load in content and style image
content = load_image('images/jesse.jpg').to(device)
# Resize style to match content, makes code easier
style = load_image('images/kahlo.jpg', shape=content.shape[-2:]).to(device)


# helper function for un-normalizing an image 
# and converting it from a Tensor image to a NumPy image for display
def im_convert(tensor):
    """ Display a tensor as an image. """
    
    image = tensor.to("cpu").clone().detach()
    image = image.numpy().squeeze()
    image = image.transpose(1,2,0)
    image = image * np.array((0.229, 0.224, 0.225)) + np.array((0.485, 0.456, 0.406))
    image = image.clip(0, 1)

    return image

# display the images
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
# content and style ims side-by-side
ax1.imshow(im_convert(content))
ax2.imshow(im_convert(style))


# ---
# ## VGG19 Layers
# 
# To get the content and style representations of an image, I have to pass an image forward throug the VGG19 network until I get to the desired layer(s) and then get the output from that layer.

# print out VGG19 structure so I can see the names of various layers
print(vgg)


# ## Content and Style Features
# 
# The code below completes the mapping of layer names to the names found in the paper for the _content representation_ and the _style representation_.


def get_features(image, model, layers=None):
    """ Run an image forward through a model and get the features for 
        a set of layers. Default layers are for VGGNet matching Gatys et al (2016)
    """
    
    ## Below, I am apping layer names of PyTorch's VGGNet to names from the paper
    ## I need the layers for the content and style representations of an image
    if layers is None:
        layers = {'0': 'conv1_1',
                  '5': 'conv2_1', 
                  '10': 'conv3_1', 
                  '19': 'conv4_1',
                  '21': 'conv4_2',  ## content representation
                  '28': 'conv5_1'}
        
    features = {}
    x = image
    # model._modules is a dictionary holding each module in the model
    for name, layer in model._modules.items():
        x = layer(x)
        if name in layers:
            features[layers[name]] = x
            
    return features


# ---
# ## Gram Matrix 
# 
# The output of every convolutional layer is a Tensor with dimensions associated with the `batch_size`, a depth, `d` and some height and width (`h`, `w`). The Gram matrix of a convolutional layer can be calculated as follows:
# * Get the depth, height, and width of a tensor using `batch_size, d, h, w = tensor.size`
# * Reshape that tensor so that the spatial dimensions are flattened
# * Calculate the gram matrix by multiplying the reshaped tensor by it's transpose 
# 
# *Note: You can multiply two matrices using `torch.mm(matrix1, matrix2)`.*

def gram_matrix(tensor):
    """ Calculate the Gram Matrix of a given tensor 
        Gram Matrix: https://en.wikipedia.org/wiki/Gramian_matrix
    """
    
    # get the batch_size, depth, height, and width of the Tensor
    _, d, h, w = tensor.size()
    
    # reshape so i'm multiplying the features for each channel
    tensor = tensor.view(d, h * w)
    
    # calculate the gram matrix
    gram = torch.mm(tensor, tensor.t())
    
    return gram 


# ## Putting it all Together
# 
# Now that i've written functions for extracting features and computing the gram matrix of a given convolutional layer; it's time to put all those pieces together! I'll extract my features from my images and calculate the gram matrices for each layer in my style representation.


# get content and style features only once before training
content_features = get_features(content, vgg)
style_features = get_features(style, vgg)

# calculate the gram matrices for each layer of my style representation
style_grams = {layer: gram_matrix(style_features[layer]) for layer in style_features}

# create a third "target" image and prep it for change
# it is a good idea to start of with the target as a copy of my *content* image
# then iteratively change its style
target = content.clone().requires_grad_(True).to(device)


# ---
# ## Loss and Weights
# 
# #### Individual Layer Style Weights
#

# 
# #### Content and Style Weight
# 
# Just like in the paper, I define an alpha (`content_weight`) and a beta (`style_weight`). This ratio will affect how _stylized_ my final image is. It's recommended I leave the content_weight = 1 and set the style_weight to achieve the ratio I want.

# weights for each style layer 
# weighting earlier layers more will result in *larger* style artifacts
# notice I am excluding `conv4_2` my content representation
style_weights = {'conv1_1': 1.,
                 'conv2_1': 0.75,
                 'conv3_1': 0.2,
                 'conv4_1': 0.2,
                 'conv5_1': 0.2}

content_weight = 1  # alpha
style_weight = 1e6  # beta


# ## Updating the Target & Calculating Losses
# #### Content Loss
# 
# The content loss will is the mean squared difference between the target and content features at layer `conv4_2`. This can be calculated as follows:
# ```
# content_loss = torch.mean((target_features['conv4_2'] - content_features['conv4_2'])**2)
# ```
# 
# #### Style Loss
# 
# The style loss is calculated in a similar way, only I have to iterate through a number of layers, specified by name in my dictionary `style_weights`.
# > I'll calculate the gram matrix for the target image, `target_gram` and style image `style_gram` at each of those layers and compare those gram matrices, calculating the `layer_style_loss`.
# > Later, this value is normalized by the size of the layer.
# 
# #### Total Loss
# 
# Finally, I'll create the total loss by adding up the style and content losses and weighting them with my specified alpha and beta!
# 
# Intermittently, I'll print out this loss;


# for displaying the target image, intermittently
show_every = 400

# iteration hyperparameters
optimizer = optim.Adam([target], lr=0.003)
steps = 4000  # here I can decide how many iterations to update my image (5000)

for ii in range(1, steps+1):
    
    # get the features from my target image
    target_features = get_features(target, vgg)
    
    # the content loss
    content_loss = torch.mean((target_features['conv4_2'] - content_features['conv4_2'])**2)
    
    # the style loss
    # initialize the style loss to 0
    style_loss = 0
    # then add to it for each layer's gram matrix loss
    for layer in style_weights:
        # get the "target" style representation for the layer
        target_feature = target_features[layer]
        target_gram = gram_matrix(target_feature)
        _, d, h, w = target_feature.shape
        # get the "style" style representation
        style_gram = style_grams[layer]
        # the style loss for one layer, weighted appropriately
        layer_style_loss = style_weights[layer] * torch.mean((target_gram - style_gram)**2)
        # add to the style loss
        style_loss += layer_style_loss / (d * h * w)
        
    # calculate the *total* loss
    total_loss = content_weight * content_loss + style_weight * style_loss
    
    # update my target image
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()
    
    # display intermediate images and print the loss
    if  ii % show_every == 0:
        print('Total loss: ', total_loss.item())
        plt.imshow(im_convert(target))
        plt.show()


# ## Display the Target Image

# display content and final, target image
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
ax1.imshow(im_convert(content))
ax2.imshow(im_convert(target))
