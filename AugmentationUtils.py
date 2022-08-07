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

    for i in range(1, len(images)):
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
    
###########################################################


def augment_image(image, how_many, probability):
    '''
    Augmentates the received image and returns it 
    '''
    #Image dimension
    ig = image
    print("ANTES DE AUMENTAR",ig.dtype)
    image = tensor_to_PILImage(image)
    w = image.width
    h = image.height
    print(w,h)
    
    #Compose class receives a list with augmentations and it returns the transformation function
    transform = A.Compose(
        [
            A.Resize(width=w, height=h),
            A.RandomCrop(width=w, height=h), #recortar pero que sea menor a la imagen este si
            #A.RandomBrightnessContrast(p=probability), #con 20% de probabilidad cambiara el brillo y contraste
            A.Rotate(limit=40, p=probability),# border_mode=cv2.BORDER_CONSTANT, #con 90% de probabilidad se va a rotar
            A.HorizontalFlip(p=probability), #voltear horizontal con probabilidad de 50%  NO
            A.VerticalFlip(p=probability), #voltear vertical con probabilidad de 10%      NO
            #No afecta y brillo tampoco.
            #A.RGBShift(r_shift_limit=25, g_shift_limit=25, b_shift_limit=25, p=probability), #Cambia aleatoriamente los valores para cada canal de la imagen RGB de entrada.
            #Se escoge uno de los siguientes con probabilidad del 10%
            A.OneOf([
                A.Blur(blur_limit=3, p=probability), #Desenfocar con tamaño de nucleo a desenforcar y probabilidad
                #A.ColorJitter(p=0.5), #Fluctuar color (El brillo, el contraste y la saturación)
            ], p=probability),
        ]
    )

    
    #Convert to numpy arry
    image = np.array(image) 
    images_list = []

    for i in range(how_many):
        #The arg is the image to be trasformed
        augmentations = transform(image=image)
        #transform returns a dictionary and the image is in the key image
        augmented_img = augmentations["image"] 
        images_list.append(augmented_img) 

    #plot_examples(images_list) 
    #returns the image in a tensor 
    imageTensor = numpyArray_to_tensor(images_list[0])
    imageTensor.resize_(images_list[0].shape[2], images_list[0].shape[1], images_list[0].shape[0])

    return imageTensor