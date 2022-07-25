import os

from cv2 import threshold
from DatasetBatch import DatasetBatch
from DatasetFactory import DatasetFactory
from DatasetsEnum import DatasetsEnum
from DatasetBatchExtractor import DatasetBatchExtractor
from OODScores.MahalanobisScore import MahalanobisScore
from torchvision.utils import save_image
from pathlib import Path
from OODScores.ScoreDelegate import ScoreDelegate
from TransferFunctions.PercentageTransferFunction import PercentageTransferFunction
from TransferFunctions.TransferFunction import TransferFunction

def augmentate_images(augmentations_probabilities, batch:DatasetBatch, destination_folder:Path):
    destination_folder.mkdir(parents=True, exist_ok=True)

    for i in range(0, 10):
        os.mkdir(os.path.join(destination_folder, str(i)))

    category = 0
    for image_index in range(len(augmentations_probabilities)):
        if (augmentations_probabilities[image_index] > 0):
            ## I need to augmentate the image

            if category == 10:
                category = 0

            #Randomly save the image in each category
            image_fullname = os.path.join(destination_folder, str(category), str(image_index) + ".png")
            save_image(batch.images[image_index], image_fullname)
            category += 1

def CreateExperiment(test_batch:DatasetBatch, batch_quantity:int, labeled_dataset, unlabeled_dataset, score:ScoreDelegate, transfer_function:TransferFunction, destination_folder, batch_size_unlabeled, ood_percentage:float):
    for current_batch in range(0, batch_quantity):
        print("Current batch: ", current_batch)
        unlabeled_batch = DatasetBatchExtractor.get_mix_batch(labeled_dataset, unlabeled_dataset, batch_size_unlabeled, ood_percentage)
        print("Unlabeled images shape(#images, channels, x, y): ", unlabeled_batch.images.shape)
        scores_for_batch = score.score_batch(unlabeled_batch)
        augmentation_probabilities = transfer_function.filter_batch(scores_for_batch)
        batch_path = os.path.join(destination_folder, "batch_" + str(current_batch))
        augmentate_images(augmentation_probabilities, unlabeled_batch, Path(os.path.join(batch_path, "train")))

        save_labeled_batch(test_batch, os.path.join(batch_path, "test"))

def GenerateLabeledBatches(batch_quantity:int, destination_folder, dataset_path, batch_size:int):
    dataset_factory = DatasetFactory(dataset_path)
    dataset_train = dataset_factory.create_training_dataset(DatasetsEnum.MNIST)                                                                                
    destination_folder = Path(os.path.join(destination_folder, "labeled"))
    
    test_batch = DatasetBatchExtractor.get_balance_batch(dataset_train, batch_size)
    for batch_index in range(batch_quantity):
        train_batch = DatasetBatchExtractor.get_balance_batch(dataset_train, batch_size)
        batch_path = os.path.join(destination_folder, "batch_" + str(batch_index))
        
        save_labeled_batch(train_batch, os.path.join(batch_path, "train"))
        save_labeled_batch(test_batch, os.path.join(batch_path, "test"))

def save_labeled_batch(train_batch:DatasetBatch, batch_path):
    for i in range(10):
        Path(os.path.join(batch_path, str(i))).mkdir(parents=True, exist_ok=True)

    for image_index in range(train_batch.size):
        save_image(train_batch.images[image_index], os.path.join(batch_path, str(train_batch.labels[image_index].item()), str(image_index) + ".png"))

# Experiment factors ------------------------------------------
labeled_datasets = [DatasetsEnum.MNIST]
unlabeled_datasets = [DatasetsEnum.SALTANDPEPPER, DatasetsEnum.GaussianNoise]
number_images = [60, 100]
threshold = [True, False]
ood_percentages = [0.5, 1] #Out of distribution percentages
augmentation_probabilities = [0.5, 1]
# Experiment factors ------------------------------------------

def main():
    # Parameters ------------------------------------------
    batch_size_labeled = 60
    batch_quantity = 2
    datasets_path = "C:\\Users\\Barnum\\Desktop\\datasets"
    destination_folder = "C:\\Users\\Barnum\\Desktop\\experiments"
    # Parameters ------------------------------------------

    #GenerateLabeledBatches(batch_quantity, destination_folder, datasets_path)

    factory = DatasetFactory(datasets_path)
    for labeled_dataset_name in labeled_datasets:
        labeled_dataset = factory.create_training_dataset(labeled_dataset_name)
        labeled_batch = DatasetBatchExtractor.get_random_batch(labeled_dataset, batch_size_labeled)
        print("Labeled images shape(#images, channels, x, y): ", labeled_batch.images.shape)
        mahanobis_score = MahalanobisScore(labeled_batch)
        for batch_size_unlabeled in number_images:
            # All 
            test_batch = DatasetBatchExtractor.get_random_batch(labeled_dataset, batch_size_unlabeled)

            for unlabeled_dataset_name in unlabeled_datasets:
                unlabeled_dataset = factory.create_unlabeled_dataset(unlabeled_dataset_name)        
                for inverse_transfer_function in threshold:
                    for ood_percentage in ood_percentages:
                        for augmentation_probability in augmentation_probabilities:
                            if(inverse_transfer_function):
                                inverse = "positive"
                            else:
                                inverse = "negative"
                            experiment_path = os.path.join(destination_folder, "unlabeled", labeled_dataset_name.value + "_" + unlabeled_dataset_name.value + "_ood"+ str(ood_percentage)+"_"+inverse+"_ap"+ str(augmentation_probability) + "_images" + str(batch_size_unlabeled))

                            transfer_function = PercentageTransferFunction(0.65, inverse_transfer_function)
                            CreateExperiment(test_batch, batch_quantity, labeled_dataset, unlabeled_dataset, mahanobis_score, transfer_function, experiment_path, batch_size_unlabeled, ood_percentage)        

if __name__ == "__main__":
   main()