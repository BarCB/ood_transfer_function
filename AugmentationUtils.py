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
            img = visualize_bbox(images[i - 1], bboxes[i - 1], class_name="Cat")
        else:
            img = images[i-1]
        fig.add_subplot(rows, columns, i)
        plt.imshow(img)
    plt.show()

# From https://albumentations.ai/docs/examples/example_bboxes/
def visualize_bbox(img, bbox, class_name, color=(255, 0, 0), thickness=5):
    '''
    Visualiza un solo cuadro delimitador en la imagen
    '''
    x_min, y_min, x_max, y_max = map(int, bbox)
    cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color, thickness)
    return img


def augment_image(image, how_many, probability):
    #Image dimension
    w = image.width 
    h = image.height 
    print(w,h)
    '''
    A la clase compose se le pasa una lista de aumentos y devuelve una funcion 
    de transformacion 
    '''
    transform = A.Compose(
        [
            A.Resize(width=w, height=h),
            #A.RandomCrop(width=1080, height=720), #recortar pero que sea menor a la imagen 
            A.RandomBrightnessContrast(p=probability), #con 20% de probabilidad cambiara el brillo y contraste
            A.Rotate(limit=40, p=probability),# border_mode=cv2.BORDER_CONSTANT, #con 90% de probabilidad se va a rotar
            A.HorizontalFlip(p=probability), #voltear horizontal con probabilidad de 50%
            A.VerticalFlip(p=probability), #voltear vertical con probabilidad de 10%
            A.RGBShift(r_shift_limit=25, g_shift_limit=25, b_shift_limit=25, p=probability), #Cambia aleatoriamente los valores para cada canal de la imagen RGB de entrada.
            #Se escoge uno de los siguientes con probabilidad del 10%
            A.OneOf([
                A.Blur(blur_limit=3, p=probability), #Desenfocar con tamaño de nucleo a desenforcar y probabilidad
                #A.ColorJitter(p=0.5), #Fluctuar color (El brillo, el contraste y la saturación)
            ], p=probability),
        ]
    )


    images_list = [image]
    #Convert to numpy arry
    image = np.array(image) 
    for i in range(how_many):
        #The arg is the image to be trasformed
        augmentations = transform(image=image)
        #transform returns a dictionary and the image is in the key image
        augmented_img = augmentations["image"] 
        images_list.append(augmented_img) 
    #Visualize images
    plot_examples(images_list) 