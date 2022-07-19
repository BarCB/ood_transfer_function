import os
from DatasetFactory import DatasetFactory
from DatasetsEnum import DatasetsEnum
from FeatureExtractor import FeatureExtractor
from DatasetSampleExtractor import DatasetSampleExtractor
from OODScores.MahalanobisScore import MahalanobisScore
from torchvision.utils import save_image
from pathlib import Path

def get_threshold(score_per_image, percent_to_filter):
    """
    Get the threshold according to the list of observations and the percent of data to filter
    :param percent_to_filter: value from 0 to 1
    :return: the threshold
    """
    copied_list = score_per_image.copy()
    copied_list.sort()
    num_to_filter = int(len(copied_list) * percent_to_filter)
    threshold = copied_list[num_to_filter]
    return threshold

def transfer_function(threshold : int, scores_for_batch, images, current_batch, destination_folder, inverse_transfer_function:bool):
    path = Path(os.path.join(destination_folder, "batch_" + str(current_batch), "train"))
    path.mkdir(parents=True, exist_ok=True)

    for i in range(0, 10):
        os.mkdir(os.path.join(path, str(i)))

    category = 0
    for image_index in range(len(scores_for_batch)):
        if ((not inverse_transfer_function) and scores_for_batch[image_index] <= threshold) or (inverse_transfer_function and scores_for_batch[image_index] > threshold):
            if category == 10:
                category = 0
            
            image_fullname = os.path.join(path, str(category), str(image_index) + ".png")
            category += 1
            save_image(images[image_index], image_fullname)

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
    for current_batch in range(0, batch_quantity):
        print("Current batch: ", current_batch)
        
        unlabeled_batch = DatasetSampleExtractor.get_random_batch(unlabeled_dataset, batch_size_unlabeled)
        print("Unlabeled images shape(#images, channels, x, y): ", unlabeled_batch.images.shape)

        scores_for_batch = mahanobis_score.score_batch(unlabeled_batch)
        threshold = get_threshold(scores_for_batch, 0.65)
        
        print("Threshold for the batch: ", threshold)
        transfer_function(threshold, scores_for_batch, unlabeled_batch.images, current_batch, destination_folder, inverse_transfer_function)

if __name__ == "__main__":
   main()
