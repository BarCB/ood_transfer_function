import random
import cv2
from matplotlib import pyplot as plt
import matplotlib.patches as patches
import numpy as np
import albumentations as A
import cv2
import albumentations as A
import numpy as np
from PIL import Image #PIL es para la manipulacion de imagenes
                      #Opencv2 ya trae manejo de imagenes 
import torch
import torchvision.transforms as T

def visualize(image):
    plt.figure(figsize=(10, 10))
    plt.axis('off')
    plt.imshow(image)
    plt.show()

def plot_examples(images, bboxes=None):
    fig = plt.figure(figsize=(15, 15))
    columns = 4
    rows = 5

    for i in range(1, len(images)+1):
        if bboxes is not None:
            img = visualize_bbox(images[i - 1], bboxes[i - 1])
        else:
            img = images[i-1]
        fig.add_subplot(rows, columns, i)
        plt.imshow(img)
    plt.show()


def visualize_bbox(img, bbox, class_name, color=(255, 0, 0), thickness=5):
    '''
    Visualiza un solo cuadro delimitador en la imagen
    '''
    x_min, y_min, x_max, y_max = map(int, bbox)
    cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color, thickness)
    return img

############This could be better in Utils ##################
def numpyArray_to_tensor(imgArray) -> torch.Tensor:
    return torch.Tensor(imgArray)

def tensor_to_PILImage(imgTensor) -> Image:
    # define a transform to convert a tensor to PIL image
    transform = T.ToPILImage()
    return transform(imgTensor)

def resizeTensor(tensor) -> torch.Tensor:
    return tensor.resize_(tensor.shape[2], tensor.shape[1], tensor.shape[0])

###########################################################


def augment_image(image, how_many, probability):
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

    image = np.array(image) 
    #The arg is the image to be trasformed
    augmentations = transform(image=image)
    #transform returns a dictionary and the image is in the key image
    augmented_img = augmentations["image"] 
    augmented_img = numpyArray_to_tensor(augmented_img)
    augmented_img = resizeTensor(augmented_img)

    return augmented_img