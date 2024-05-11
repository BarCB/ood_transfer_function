from pathlib import Path
import albumentations as A
import AugmentationUtils as AT
from Batches.DatasetBatch import DatasetBatch
from Batches.DatasetBatchExtractor import DatasetBatchExtractor
from DatasetFactory import DatasetFactory
from DatasetsEnum import DatasetsEnum
from PIL import Image

def augmentate_images(batch:DatasetBatch, destination_folder:Path, number_categories):

    
    for i in range(0, number_categories):
        Path(destination_folder, str(i)).mkdir(parents=True)
    for image_index in range(batch.size):
        augmented_image = AT.apply_salt_n_pepper_noise(batch.getImages()[image_index]) 
    
        image_fullname = Path(destination_folder, str(batch.labels[image_index].item()))

        pil_image = Image.fromarray(augmented_image)
        pil_image.save(Path(image_fullname, f"{str(image_index)}_augmented.png"))

if __name__ == "__main__":
    
    output_folder = "C:\\Users\\Barnum\\Desktop\\datasets"
    datasets_path = "C:\\Users\\Barnum\\Desktop\\datasets"
    factory = DatasetFactory(datasets_path)
    dataset = DatasetsEnum.Indiana
    source_dataset = factory.create_dataset_original_size(dataset)
    
    batch = DatasetBatchExtractor.get_max_batch(source_dataset)
    
    number_categories = len(source_dataset.class_to_idx)
    augmentate_images(batch, Path(output_folder, dataset.value + "SnP", "all") , number_categories)