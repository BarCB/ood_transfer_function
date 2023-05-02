import torch
import numpy as np
from typing import List
from FeatureExtractor import FeatureExtractor
from CudaDeviceSingleton import CudaDeviceSingleton
from OODScores.ScoreDelegate import ScoreDelegate
from Batches.DatasetBatch import DatasetBatch

class MahalanobisScore(ScoreDelegate):
    def __init__(self, labeled_batch:DatasetBatch) -> None:
        self.feature_extractor = FeatureExtractor()
        self.pseudoinverse_covariance_matrix, self.labeled_features_mean = self.__calculate_covariance_matrix_pseudoinverse(labeled_batch)
        super().__init__()

    def score_batch(self, unlabeled_batch: DatasetBatch) -> List[int]:
        #Extrats feature for each unlabeled image
        gauss_likelihoods_final_all_obs = []
        for image_index in range(0, unlabeled_batch.size):
            image = unlabeled_batch.images[image_index:image_index+1, :, :, :]
            features_bunch = self.feature_extractor.extract_features(image)
            likelihoods_gauss_batch = self.__calculate_mahalanobis_distance(features_bunch)
            gauss_likelihoods_final_all_obs += [likelihoods_gauss_batch]
            
        return gauss_likelihoods_final_all_obs

    def __calculate_mahalanobis_distance(self, features_obs_batch) -> int:
        """
        Calculate the Mahalanbis distance for a features obs batch of 1
        :param Covariance_mat:
        :param features_obs_batch: Only 1 observation is supported
        :param mean_features_all_observations:
        :return:
        """
        # substract the mean to the current observation
        fac_quad_form = features_obs_batch - self.labeled_features_mean
        fac_quad_form_t = fac_quad_form.transpose(0, 1)
        # evaluate likelihood for all observations
        likelihood_batch = fac_quad_form.mm(self.pseudoinverse_covariance_matrix).mm(fac_quad_form_t).item()
        return likelihood_batch

    def __calculate_covariance_matrix_pseudoinverse(self, image_batch:DatasetBatch) -> tuple[torch.tensor, torch.tensor]:
        """
        Calculates the pseudo inverse covariance matrix and the mean of the all the features extracted from image_batch
        :param image_batch: the batch of images from with the matrix and the mean are calculated
        :return: pseudo inverse covariance matrix, mean of features
        """
        tensor_bunch = self.feature_extractor.extract_features(image_batch.images)
        
        features_all_observations = tensor_bunch[:, :].cpu().detach().numpy()

        features_all_observations = features_all_observations.transpose()
        device = CudaDeviceSingleton().get_device()
        mean_features_all_observations = torch.mean(torch.tensor(features_all_observations, device=device, dtype = torch.float), 1)
        
        #calculate the covariance matrix
        covariance_matrix = np.cov(features_all_observations)
        covariance_matrix_pinverse = torch.tensor(covariance_matrix, device=device, dtype=torch.float)
        
        #return the cov mat as a tensor
        return covariance_matrix_pinverse, mean_features_all_observations