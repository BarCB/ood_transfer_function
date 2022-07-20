import os
from DatasetBatch import DatasetBatch
from DatasetFactory import DatasetFactory
from DatasetsEnum import DatasetsEnum
from FeatureExtractor import FeatureExtractor
from DatasetSampleExtractor import DatasetSampleExtractor
from OODScores.MahalanobisScore import MahalanobisScore
from torchvision.utils import save_image
from pathlib import Path
from TransferFunctions.PercentageTransferFunction import PercentageTransferFunction

def augmentate_images(images_to_augmentate, batch:DatasetBatch, destination_folder):
    path = Path(os.path.join(destination_folder, "all"))
    path.mkdir(parents=True, exist_ok=True)

    for image_index in range(len(images_to_augmentate)):
        if (images_to_augmentate[image_index]):
            ## I need to augmentate the image
            image_fullname = os.path.join(path, str(image_index) + ".png")
            save_image(batch.images[image_index], image_fullname)

def main():
    batch_size_labeled = 60
    batch_size_unlabeled = 60
    batch_quantity = 1
    unlabeled_dataset_name = DatasetsEnum.GaussianNoise
    inverse_transfer_function = True

    factory = DatasetFactory("C:\\Users\\Barnum\\Desktop\\datasets")
    labeled_dataset = factory.create_training_dataset(DatasetsEnum.MNIST)
    unlabeled_dataset = factory.create_unlabeled_dataset(unlabeled_dataset_name)
    if(not inverse_transfer_function):
        destination_folder = "C:\\Users\\Barnum\\Desktop\\experiments\\" + unlabeled_dataset_name.value + "_positive_p" + "_from" + str(batch_size_unlabeled) + "Images"
    else:
        destination_folder = "C:\\Users\\Barnum\\Desktop\\experiments\\" + unlabeled_dataset_name.value + "_negative_p" + "_from" + str(batch_size_unlabeled) + "Images"

    labeled_batch = DatasetSampleExtractor.get_random_batch(labeled_dataset, batch_size_labeled)
    print("Labeled images shape(#images, channels, x, y): ", labeled_batch.images.shape)
    
    mahanobis_score = MahalanobisScore(labeled_batch)
    transfer_function = PercentageTransferFunction(0.65, inverse_transfer_function)
    for current_batch in range(0, batch_quantity):
        print("Current batch: ", current_batch)
        
        unlabeled_batch = DatasetSampleExtractor.get_random_batch(unlabeled_dataset, batch_size_unlabeled)
        print("Unlabeled images shape(#images, channels, x, y): ", unlabeled_batch.images.shape)

        scores_for_batch = mahanobis_score.score_batch(unlabeled_batch)

        images_to_augmentate = transfer_function.filter_batch(scores_for_batch)
        
        augmentate_images(images_to_augmentate, unlabeled_batch, destination_folder)
        
        

if __name__ == "__main__":
   main()
