import torch
import torchvision.models as models
from fastai.vision.all import *
from fastai.vision.data import *

class FeatureExtractor:
  def __init__(self):
    self.model = models.alexnet(pretrained=True)
    self.feature_extractor = self.__create_feature_extractor()

  def __create_feature_extractor(self):
      """
      Gets a feature extractor from a model
      param model: is the fastAI model
      """
      path = untar_data(URLs.MNIST_SAMPLE)
      data = ImageDataLoaders.from_folder(path)
      # save learner to reload it as a pytorch model
      learner = Learner(data, self.model, metrics=[accuracy])
      path = os.path.join(os.getcwd(),"final_model_bah.pk")
      learner.export(path)
      torch_dict = torch.load(path)
      # get the model
      model_loaded = torch_dict.model
      model_loaded.eval()
      # put it on gpu!
      device = 'cuda' if torch.cuda.is_available() else 'cpu'
      model_loaded = model_loaded.to(device=device)
      # usually the last set of layers act as classifier, therefore we discard it    
      feature_extractor = model_loaded.features
      return feature_extractor
  
  def extract_feature_bunch(self, images):
    features_bunch = self.feature_extractor(images)
    print("Extracted features(#images, #featureMatrixes, x(featureMatrixes), y(featureMatrixes)):")
    print(features_bunch.shape)
    return features_bunch
  
  def get_batch_features(self, tensorbunch_unlabeled, batch_size_unlabeled, batch_number):
    """
    Get the batch of features using a specific feature extractor
    :param tensorbunch_unlabeled: tensorbunch to evaluate using the feature extractor
    :param batch_size_unlabeled: batch size to use during evaluation
    :param batch_number: batch number to evaluate
    :return: features extracted
    """

    # create the batch of tensors to get its features
    batch_tensors1 = tensorbunch_unlabeled[
                      batch_number * batch_size_unlabeled:(batch_number + 1) * batch_size_unlabeled, :, :, :]
    # batch indices for accountability
    batch_indices = torch.arange(batch_number * batch_size_unlabeled, (batch_number + 1) * batch_size_unlabeled)
    # print("batch tensors ", batch_tensors1.shape)
    # get the  features from the selected batch
    features_bunch1 = self.extract_features(batch_tensors1)
    # get the values of a specific dimension
    # values_dimension_bunch1 = features_bunch1[:, :].cpu().detach().numpy()
    values_dimension_bunch1 = features_bunch1[:, :]
    return values_dimension_bunch1, batch_indices

  def extract_features(self, batch_tensors1):
    """
    Extract features from a tensor bunch
    :param batch_tensors1:
    :return:
    """
    
    features_bunch1 = self.feature_extractor(batch_tensors1)
    # pool of non-square window
    #print("features_bunch1 shape ", features_bunch1.shape)
    avg_layer = nn.AvgPool2d((features_bunch1.shape[2], features_bunch1.shape[3]), stride=(1, 1))
    #averaging the features to lower the dimensionality in case is not wide resnet
    features_bunch1 = avg_layer(features_bunch1)
    features_bunch1 = features_bunch1.view(-1, features_bunch1.shape[1] * features_bunch1.shape[2] * features_bunch1.shape[3])

    return features_bunch1