import shutil
from Batches.DatasetBatch import DatasetBatch
from DatasetFactory import DatasetFactory
from DatasetsEnum import DatasetsEnum
from Batches.DatasetBatchExtractor import DatasetBatchExtractor
from OODScores.MahalanobisScore import MahalanobisScore
from torchvision.utils import save_image
from pathlib import Path
from OODScores.ScoreDelegate import ScoreDelegate
from TransferFunctions.PercentageTransferFunction import PercentageTransferFunction
from TransferFunctions.TransferFunction import TransferFunction
import AugmentationUtils as AT
from PIL import Image
from TransferFunctions.TransferFunctionEnum import TransferFunctionEnum
from TransferFunctions.TransferFunctionFactory import TransferFunctionFactory

def augmentate_images(augmentations_probabilities, batch:DatasetBatch, destination_folder:Path):
    for image_index in range(len(augmentations_probabilities)):
        probability = augmentations_probabilities[image_index]
        images = AT.augment_image(batch.getImages()[image_index], probability) 
    
        pil_image = Image.fromarray(images[1])
        image_fullname = Path(destination_folder, str(batch.labels[image_index].item()))
        pil_image.save(Path(image_fullname, f"{str(image_index)}_original.png"))

        was_augmented = images[0]
        if(was_augmented):
            pil_image = Image.fromarray(images[2])
            pil_image.save(Path(image_fullname, f"{str(image_index)}_augmented.png"))

def create_destination_folder(destination_path:Path):
    if destination_path.exists() and destination_path.is_dir():
        shutil.rmtree(destination_path)
    destination_path.mkdir(parents=True)

def create_experiment(test_batch:DatasetBatch, batch_quantity:int, source_dataset, target_dataset, score:ScoreDelegate, 
                      transfer_function:TransferFunction, destination_folder:Path, batch_size_unlabeled, ood_percentage:float):
    create_destination_folder(destination_folder)
    number_categories = len(source_dataset.class_to_idx)
    for current_batch in range(0, batch_quantity):
        print("Current batch: ", current_batch)
        target_batch = DatasetBatchExtractor.get_mix_batch(source_dataset, target_dataset, batch_size_unlabeled, ood_percentage)
        print("Unlabeled images shape(#images, channels, x, y): ", target_batch.images.shape)
        scores_for_batch = score.score_batch(target_batch)
        augmentation_probabilities = transfer_function.filter_batch(scores_for_batch)
        #probabilities per image
        batch_path = Path(destination_folder, "batch_" + str(current_batch))
        train_path = Path(batch_path, "train")

        for i in range(0, number_categories):
            Path(train_path, str(i)).mkdir(parents=True)

        augmentate_images(augmentation_probabilities, target_batch, train_path)
        #save_image_batch(test_batch, Path(batch_path, "test"), number_categories)

def generate_source_batches(batch_quantity:int, destination_folder:Path, dataset_path:Path, batch_size:int, test_size:int):
    dataset_factory = DatasetFactory(dataset_path)
    dataset_train = dataset_factory.create_dataset(DatasetsEnum.MNIST)                                                                                
    destination_folder = Path(destination_folder, "source")
    
    test_batch = DatasetBatchExtractor.get_balance_batch(dataset_train, test_size)
    for batch_index in range(batch_quantity):
        #train_batch = DatasetBatchExtractor.get_balance_batch(dataset_train, batch_size)
        batch_path = Path(destination_folder, "batch_" + str(batch_index))
        
        #save_image_batch(train_batch, Path(batch_path, "train"))
        save_image_batch(test_batch, Path(batch_path, "test"))

def generate_target_batches(source_batch_size:int, batch_quantity:int, datasets_path:Path, destination_folder:Path, test_size:int):
    factory = DatasetFactory(datasets_path)
    
    for source_dataset_name in source_datasets:
        source_dataset = factory.create_dataset(source_dataset_name)
        source_batch = DatasetBatchExtractor.get_random_batch(source_dataset, source_batch_size)
        print("Labeled images shape(#images, channels, x, y): ", source_batch.images.shape)
        mahanobis_score = MahalanobisScore(source_batch)
        for batch_size_unlabeled in number_images:
            test_batch = DatasetBatchExtractor.get_random_batch(source_dataset, test_size)

            for target_dataset_name in target_datasets:
                target_dataset = factory.create_dataset(target_dataset_name)        
                for transfer_function_type in transfer_functions:
                    for ood_percentage in ood_percentages:
                        experiment_path = Path(destination_folder, "target", source_dataset_name.value + "_" + target_dataset_name.value + 
                                                "_ood" + str(ood_percentage)+"_" + transfer_function_type.value + "_images" + str(batch_size_unlabeled))

                        transfer_function = TransferFunctionFactory.create_transfer_function(transfer_function_type)
                        create_experiment(test_batch, batch_quantity, source_dataset, target_dataset, mahanobis_score, transfer_function, experiment_path, batch_size_unlabeled, ood_percentage)   

def save_image_batch(train_batch:DatasetBatch, batch_path:Path, number_categories:int):
    for i in range(number_categories):
        Path(batch_path, str(i)).mkdir(parents=True, exist_ok=True)

    for image_index in range(train_batch.size):
        save_image(train_batch.images[image_index], Path(batch_path, str(train_batch.labels[image_index].item()), str(image_index) + ".png"))

# Experiment factors ------------------------------------------
source_datasets = [DatasetsEnum.SVHN]
target_datasets = [DatasetsEnum.SVHN]
number_images = [200]
transfer_functions = [TransferFunctionEnum.NoneFunction]
ood_percentages = [0] #Out of distribution percentages
# Experiment factors ------------------------------------------

def main():
    # Parameters ------------------------------------------
    source_batch_size = 10000  #MNIST has 42k images but for hardware capacity 25000 is used
    batch_quantity = 1
    datasets_path = "C:\\Users\\Barnum\\Desktop\\datasets"
    destination_folder = "C:\\Users\\Barnum\\Desktop\\experiments6"
    test_size = 80
    # Parameters ------------------------------------------

    #generate_source_batches(batch_quantity, destination_folder, datasets_path, source_batch_size, test_size)

    generate_target_batches(source_batch_size, batch_quantity, datasets_path, destination_folder, test_size) 

if __name__ == "__main__":
   main()