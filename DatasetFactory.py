import torchvision
import torchvision.transforms as transforms
from DatasetsEnum import DatasetsEnum
from pathlib import Path

class DatasetFactory:
    def __init__(self, datasets_path:Path):
        self.datasets_path = datasets_path

    def create_training_dataset(self, dataset_type:DatasetsEnum):
        labeled_path = Path(self.datasets_path, dataset_type.value, "all")
        dataset =  torchvision.datasets.ImageFolder(labeled_path, transform = self.__get_transformation())
        print("Training dataset created")   
        print(dataset)
        return dataset

    def create_unlabeled_dataset(self, augmented_dataset_type:DatasetsEnum):
        datasetPath = Path(self.datasets_path, augmented_dataset_type.value, "all")    
        dataset = torchvision.datasets.ImageFolder(datasetPath, transform = self.__get_transformation())
        print("Unlabeled dataset created")   
        print(dataset)
        return dataset

    def __get_transformation(self):
        return transforms.Compose([
            transforms.Resize((63,63)),
            transforms.ToTensor(),
            ])