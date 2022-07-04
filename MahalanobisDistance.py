import torch
import numpy as np
from FeatureExtractor import FeatureExtractor
from CudaDeviceSingleton import CudaDeviceSingleton

def calculate_Mahalanobis_distance(Covariance_mat_pseudoinverse, features_obs_batch, mean_features_all_observations):
    """
    Calculate the Mahalanbis distance for a features obs batch of 1
    :param Covariance_mat:
    :param features_obs_batch: Only 1 observation is supported
    :param mean_features_all_observations:
    :return:
    """
    # substract the mean to the current observation
    fac_quad_form = features_obs_batch - mean_features_all_observations
    fac_quad_form_t = fac_quad_form.transpose(0, 1)
    # evaluate likelihood for all observations
    likelihood_batch = fac_quad_form.mm(Covariance_mat_pseudoinverse).mm(fac_quad_form_t).item()
    return likelihood_batch

def calculate_covariance_matrix_pseudoinverse(tensor_bunch, feature_extractor: FeatureExtractor, dimensions: int, batch_size: int = 5):
  """
  Returns the pseudo inverse of the cov matrix
  :param tensor_bunch: 
  :param feature_extractor:
  :param dimensions: dimensions of WHAT
  :param batch_size:
  :param num_bins:
  :param plot:
  :return:
  """
  total_number_obs_1 = tensor_bunch.shape[0]
  print("Number of observations", total_number_obs_1)
  print("Number of dimensions ", dimensions)
  device = CudaDeviceSingleton().get_device()
  
  features_all_observations = torch.zeros((total_number_obs_1, dimensions), device=device)

  # print("total number of obs ", total_number_obs_1)
  number_batches = total_number_obs_1 // batch_size
  batch_tensors1 = tensor_bunch[0: batch_size, :, :, :]
  # get the  features from the selected batch
  features_bunch1 = feature_extractor.extract_features(batch_tensors1)
  # for each dimension, calculate its histogram
  # get the values of a specific dimension
  features_all_observations = features_bunch1[:, :].cpu().detach().numpy()
  # Go through each batch...
  
  for current_batch_num in range(1, number_batches):
      # create the batch of tensors to get its features
      batch_tensors1 = tensor_bunch[(current_batch_num) * batch_size: (current_batch_num + 1) * batch_size, :, :,:]
      # get the  features from the selected batch
      features_bunch1 = feature_extractor.extract_features(batch_tensors1)
      # get the values of a specific dimension
      values_dimension_bunch1 = features_bunch1[:, :].cpu().detach().numpy()
      features_all_observations = np.concatenate((features_all_observations, values_dimension_bunch1), 0)
  features_all_observations = features_all_observations.transpose()
  mean_features_all_observations = torch.mean(torch.tensor(features_all_observations, device=device, dtype = torch.float), 1)
  #calculate the covariance matrix
  Covariance_mat = np.cov(features_all_observations)
  Covariance_matrix_pinv = torch.tensor(Covariance_mat, dtype=torch.float, device=device)
  #return the cov mat as a tensor
  Covariance_matrix_pinv_torch = torch.tensor(Covariance_matrix_pinv, device=device, dtype = torch.float)


  return Covariance_matrix_pinv_torch, mean_features_all_observations