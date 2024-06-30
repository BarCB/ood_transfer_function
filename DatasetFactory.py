import torchvision
import torchvision.transforms as transforms
from DatasetsEnum import DatasetsEnum
from pathlib import Path

class DatasetFactory:
    def __init__(self, datasets_path:Path):
        self.datasets_path = datasets_path

    def create_dataset(self, dataset_type:DatasetsEnum):
        dataset_path = Path(self.datasets_path, dataset_type.value, "all")
        transformation = self.__get_transformation()
        if dataset_type == DatasetsEnum.MNIST:
            transformation = self.__get_transformation_for_grey()

        dataset =  torchvision.datasets.ImageFolder(dataset_path, transform = transformation)
        print("Dataset created")   
        print(dataset)
        return dataset
    
    def create_dataset_original_size(self, dataset_type:DatasetsEnum):
        dataset_path = Path(self.datasets_path, dataset_type.value, "all")
        transformation = self.__get_transformation_original_size()
        dataset =  torchvision.datasets.ImageFolder(dataset_path, transform = transformation)
        print("Dataset created")   
        print(dataset)
        return dataset

    def __get_transformation_for_grey(self):
        return transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize((63, 63)),
            transforms.ToTensor(),
        ])
    
    def __get_transformation_original_size(self):
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((2000, 2000)),
        ])

    def __get_transformation(self):
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])