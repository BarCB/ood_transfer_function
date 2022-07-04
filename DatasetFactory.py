import torch
import torchvision
import torchvision.transforms as transforms
from CudaDeviceSingleton import CudaDeviceSingleton
from DatasetsEnum import DatasetsEnum
import os
import albumentations as A
from albumentations.pytorch import ToTensorV2

from pathlib import Path

class DatasetFactory:
    def __init__(self, datasets_path:Path):
        self.datasets_path = datasets_path

    def create_training_dataset(self, dataset_type:DatasetsEnum):
        labeled_path = os.path.join(self.datasets_path, dataset_type.value, "all")
        dataset =  torchvision.datasets.ImageFolder(labeled_path, transform = self.__get_transformation())
        print("Training dataset created")   
        print(dataset)
        return dataset

    def create_unlabeled_dataset(self, augmented_dataset_type:DatasetsEnum, augmentation_probability:float):
        datasetPath = os.path.join(self.datasets_path, augmented_dataset_type.value)    
        dataset = torchvision.datasets.ImageFolder(datasetPath, transform = self.__get_transformation2(augmentation_probability))
        print("Unlabeled dataset created")   
        print(dataset)
        return dataset

    def __get_transformation(self):
        return transforms.Compose([
            transforms.Resize((120,120)),
            transforms.ToTensor(),
            ])

    def __get_transformation2(self, augmentation_probability:float):
        return transforms.Compose([
            transforms.Resize((120,120)),
            transforms.ToTensor(),
            transforms.RandomInvert(p=augmentation_probability)
            ])

    def __get_albumentation_transformation(self):
        return A.Compose([
            A.augmentations.geometric.resize.Resize(120, 120),
            A.RGBShift(r_shift_limit=25, g_shift_limit=25, b_shift_limit=25, p=0.9), #Cambia aleatoriamente los valores para cada canal de la imagen RGB de entrada.
            ToTensorV2(),
        ])