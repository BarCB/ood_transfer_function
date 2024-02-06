
import numpy as np
import albumentations as A
from PIL import Image
import torch
import torchvision.transforms as T
import random
def tensor_to_PILImage(imgTensor:torch.Tensor) -> Image:
    # define a transform to convert a tensor to PIL image
    transform = T.ToPILImage()
    return transform(imgTensor)

def augment_image(tensor_image:torch.Tensor, probability:float):
    '''
    Augmentates the received image and returns it 
    '''
    chance = random.uniform(0, 1)
    augmentate_image = False
    if(chance < probability):
        augmentate_image = True

    pil_image = tensor_to_PILImage(tensor_image)
    array_image = np.array(pil_image) 
    if(augmentate_image):
        transformer = create_transformation(pil_image)
        augmentations = transformer(image=array_image)
        augmented_image = augmentations["image"]
        return [augmentate_image, array_image, augmented_image]
    else:
        return [augmentate_image, array_image]

def create_transformation(pil_image)->A.Compose:
    height = pil_image.width
    width = pil_image.height
    
    #Compose class receives a list with augmentations and it returns the transformation function
    transform = A.Compose(
        [
            #It Chooses only one of the transformations
            A.OneOf([
                A.RandomCrop(width=width-5, height=height-5, p=1),
                A.Resize(width=width-5, height=height+5, p=1),
                A.Rotate(limit=20, p=1),
                A.Blur(blur_limit=5, p=1), 
                A.OpticalDistortion(p=1),
                A.GaussNoise (var_limit=(10, 300), p=1),
                A.Emboss (alpha=(0.2, 0.5), strength=(0.2, 0.7), p=1),
                A.PixelDropout (dropout_prob=0.01, drop_value=0, p=1),
                A.Solarize (p=1)
            ])
        ], p=1
    )
    
    return transform