import shutil
from cv2 import threshold
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
    category = 0

    for image_index in range(len(augmentations_probabilities)):
        probability = augmentations_probabilities[image_index]
        if (probability > 0):
            #img is a tensor
            img = AT.augment_image(batch.getImages()[image_index], probability) 
            img = Image.fromarray(img)

            if category == 10:  
                category = 0

            #Randomly save the image in each category
            image_fullname = Path(destination_folder, str(batch.labels[image_index].item()), str(image_index) + ".png")
            img.save(image_fullname)
            category += 1

def create_destination_folder(destination_path:Path):
    if destination_path.exists() and destination_path.is_dir():
        shutil.rmtree(destination_path)
    destination_path.mkdir(parents=True)

def create_experiment(test_batch:DatasetBatch, batch_quantity:int, labeled_dataset, unlabeled_dataset, score:ScoreDelegate, 
                      transfer_function:TransferFunction, destination_folder:Path, batch_size_unlabeled, ood_percentage:float):
    create_destination_folder(destination_folder)
    for current_batch in range(0, batch_quantity):
        print("Current batch: ", current_batch)
        unlabeled_batch = DatasetBatchExtractor.get_mix_batch(labeled_dataset, unlabeled_dataset, batch_size_unlabeled, ood_percentage)
        print("Unlabeled images shape(#images, channels, x, y): ", unlabeled_batch.images.shape)
        scores_for_batch = score.score_batch(unlabeled_batch)
        augmentation_probabilities = transfer_function.filter_batch(scores_for_batch)
        #probabilities per image
        batch_path = Path(destination_folder, "batch_" + str(current_batch))
        train_path = Path(batch_path, "train")

        for i in range(0, 10):
            Path(train_path, str(i)).mkdir(parents=True)

        augmentate_images(augmentation_probabilities, unlabeled_batch, train_path)
        save_image_batch(test_batch, Path(batch_path, "test"))

def generate_labeled_batches(batch_quantity:int, destination_folder:Path, dataset_path:Path, batch_size:int, test_size:int):
    dataset_factory = DatasetFactory(dataset_path)
    dataset_train = dataset_factory.create_training_dataset(DatasetsEnum.MNIST)                                                                                
    destination_folder = Path(destination_folder, "labeled")
    
    test_batch = DatasetBatchExtractor.get_balance_batch(dataset_train, test_size)
    for batch_index in range(batch_quantity):
        train_batch = DatasetBatchExtractor.get_balance_batch(dataset_train, batch_size)
        batch_path = Path(destination_folder, "batch_" + str(batch_index))
        
        save_image_batch(train_batch, Path(batch_path, "train"))
        save_image_batch(test_batch, Path(batch_path, "test"))

def save_image_batch(train_batch:DatasetBatch, batch_path:Path):
    for i in range(10):
        Path(batch_path, str(i)).mkdir(parents=True, exist_ok=True)

    for image_index in range(train_batch.size):
        save_image(train_batch.images[image_index], Path(batch_path, str(train_batch.labels[image_index].item()), str(image_index) + ".png"))

# Experiment factors ------------------------------------------
labeled_datasets = [DatasetsEnum.MNIST]
unlabeled_datasets = [DatasetsEnum.SVHN]
number_images = [100]
transfer_functions = [TransferFunctionEnum.StepFunctionPositive]
ood_percentages = [0.5] #Out of distribution percentages
# Experiment factors ------------------------------------------

def main():
    # Parameters ------------------------------------------
    batch_size_labeled = 25000  #MNIST has 42k images but for hardware capacity 25000 is used
    batch_quantity = 10
    datasets_path = "C:\\Users\\Barnum\\Desktop\\datasets"
    destination_folder = "C:\\Users\\Barnum\\Desktop\\experiments3"
    test_size = 60
    # Parameters ------------------------------------------

    #generate_labeled_batches(batch_quantity, destination_folder, datasets_path, batch_size_labeled, test_size)

    factory = DatasetFactory(datasets_path)
    for labeled_dataset_name in labeled_datasets:
        labeled_dataset = factory.create_training_dataset(labeled_dataset_name)
        labeled_batch = DatasetBatchExtractor.get_random_batch(labeled_dataset, batch_size_labeled)
        print("Labeled images shape(#images, channels, x, y): ", labeled_batch.images.shape)
        mahanobis_score = MahalanobisScore(labeled_batch)
        for batch_size_unlabeled in number_images:
            test_batch = DatasetBatchExtractor.get_random_batch(labeled_dataset, test_size)

            for unlabeled_dataset_name in unlabeled_datasets:
                unlabeled_dataset = factory.create_unlabeled_dataset(unlabeled_dataset_name)        
                for transfer_function_type in transfer_functions:
                    for ood_percentage in ood_percentages:
                        experiment_path = Path(destination_folder, "unlabeled", labeled_dataset_name.value + "_" + unlabeled_dataset_name.value + 
                                                "_ood" + str(ood_percentage)+"_" + transfer_function_type.value + "_images" + str(batch_size_unlabeled))

                        transfer_function = TransferFunctionFactory.create_transfer_function(transfer_function_type)
                        create_experiment(test_batch, batch_quantity, labeled_dataset, unlabeled_dataset, mahanobis_score, transfer_function, experiment_path, batch_size_unlabeled, ood_percentage)        

if __name__ == "__main__":
   main()