from DatasetFactory import DatasetFactory
from DatasetsEnum import DatasetsEnum
from FeatureExtractor import FeatureExtractor
import MahalanobisDistance as md
from torchvision.utils import save_image
import os
from pathlib import Path

def score_unlabeled_batch(unlabeled_images, feature_extractor : FeatureExtractor, pseudoinverse_covariance_matrix, mean_features_all_observations):
    print("Evaluating Mahalanobis distance for unlabeled data...")
    gauss_likelihoods_final_all_obs = []
    unlabeled_num_images = unlabeled_images.shape[0]
    print("Number of unlabeled images", unlabeled_num_images)
    
    #Extrats feature for each unlabeled image
    for current_batch_num_unlabeled in range(0, unlabeled_num_images):
        values_features_bunch_unlabeled, _ = feature_extractor.get_batch_features(unlabeled_images, 1, current_batch_num_unlabeled)
        # go  through each dimension, and calculate the likelihood for the whole unlabeled dataset
        likelihoods_gauss_batch = md.calculate_Mahalanobis_distance(pseudoinverse_covariance_matrix, values_features_bunch_unlabeled, mean_features_all_observations)
        gauss_likelihoods_final_all_obs += [likelihoods_gauss_batch]
        
    #Mahalanobis distance by unlabeled image batch
    print(gauss_likelihoods_final_all_obs)
    return gauss_likelihoods_final_all_obs

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
    augmentation_probability = 0.5
    unlabeled_dataset_name = DatasetsEnum.GaussianNoise
    inverse_transfer_function = True

    factory = DatasetFactory("C:\\Users\\Barnum\\Desktop\\datasets")
    labeled_dataset = factory.create_training_dataset(DatasetsEnum.MNIST)
    unlabeled_dataset = factory.create_unlabeled_dataset(unlabeled_dataset_name, augmentation_probability)
    if(not inverse_transfer_function):
        destination_folder = "C:\\Users\\Barnum\\Desktop\\experiments\\" + unlabeled_dataset_name.value + "_positive_p"+str(augmentation_probability) + "_from" + str(batch_size_unlabeled) + "Images"
    else:
        destination_folder = "C:\\Users\\Barnum\\Desktop\\experiments\\" + unlabeled_dataset_name.value + "_negative_p"+str(augmentation_probability) + "_from" + str(batch_size_unlabeled) + "Images"

    labeled_images, _ = factory.get_random_batch(labeled_dataset, batch_size_labeled)
    print("Labeled images shape(#images, channels, x, y): ", labeled_images.shape)
    
    feature_extractor = FeatureExtractor()
    labeled_features_bunch = feature_extractor.extract_feature_bunch(labeled_images)
    features_quantity = labeled_features_bunch.shape[1]
    pseudoinverse_covariance_matrix, mean_features_all_observations = md.calculate_covariance_matrix_pseudoinverse(labeled_images, feature_extractor, features_quantity, batch_size_labeled)
    for current_batch in range(0, batch_quantity):
        print("Current batch: ", current_batch)
        
        unlabeled_images, _ = factory.get_random_batch(unlabeled_dataset, batch_size_unlabeled)
        print("Unlabeled images shape(#images, channels, x, y): ", unlabeled_images.shape)

        scores_for_batch = score_unlabeled_batch(unlabeled_images, feature_extractor,  pseudoinverse_covariance_matrix, mean_features_all_observations)

        threshold = get_threshold(scores_for_batch, 0.65)
        
        print("Threshold for the batch: ", threshold)
        transfer_function(threshold, scores_for_batch, unlabeled_images, current_batch, destination_folder, inverse_transfer_function)

if __name__ == "__main__":
   main()
