import os
import torchvision.transforms as transforms
from torchvision.utils import save_image
from pathlib import Path
from DatasetFactory import DatasetFactory
from DatasetsEnum import DatasetsEnum
    
def get_transformation():
    return transforms.Compose([
        transforms.Resize((120,120)),
        transforms.ToTensor(),
        ])


dataset_factory = DatasetFactory("C:\\Users\\Barnum\\Desktop\\datasets")

dataset_train = dataset_factory.create_training_dataset(DatasetsEnum.MNIST)                                                                                

destination_folder = Path("folder2\\train")
for i in range(10):
    images, labels = dataset_factory.get_balance_batch(dataset_train, 100)
    batch_path = os.path.join(destination_folder, "batch_" + str(i), "train")
    for i in range(10):
        Path(os.path.join(batch_path, str(i))).mkdir(parents=True, exist_ok=True)

    for image_index in range(len(images)):
        save_image(images[image_index], os.path.join(batch_path, str(labels[image_index].item()), str(image_index) + ".png"))


