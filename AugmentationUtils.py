
import numpy as np
import albumentations as A
from PIL import Image #Image manipulation
import torch
import torchvision.transforms as T


def tensor_to_PILImage(imgTensor:torch.Tensor) -> Image:
    # define a transform to convert a tensor to PIL image
    transform = T.ToPILImage()
    return transform(imgTensor)


def augment_image(image:torch.Tensor, probability:float):
    '''
    Augmentates the received image and returns it 
    '''
    #Image dimension
    image = tensor_to_PILImage(image)
    w = image.width
    h = image.height
    #Compose class receives a list with augmentations and it returns the transformation function
    transform = A.Compose(
        [
            #It Chooses only one of the transformations
            A.OneOf([
                A.RandomCrop(width=w-5, height=h-5, p=probability),
                A.Resize(width=w-5, height=h+5, p=probability),
                A.Rotate(limit=20, p=probability),
                A.Blur(blur_limit=5, p=probability), 
                A.OpticalDistortion(p=probability),
                A.GaussNoise (var_limit=(10, 300), p=probability),
                A.Emboss (alpha=(0.2, 0.5), strength=(0.2, 0.7), p=probability),
                A.PixelDropout (dropout_prob=0.01, drop_value=0, p=probability),
                A.Solarize (p=probability)
            ], p=probability),
        ]
    )
    should_augmentate = True
    image = np.array(image) 
    if(should_augmentate):
        #The arg is the image to be trasformed
        augmentations = transform(image=image)
        #transform returns a dictionary and the image is in the key image
        augmented_img = augmentations["image"]
        return augmented_img
    else:
        return image