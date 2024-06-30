import torchvision.models as models
from fastai.vision.all import *
from fastai.vision.data import *
from CudaDeviceSingleton import CudaDeviceSingleton

class FeatureExtractor:
  def __init__(self):
    self.model = models.alexnet(pretrained=True)
    self.feature_extractor = self.__create_feature_extractor()

  def __create_feature_extractor(self):
      """
      Gets a feature extractor from a model
      param model: is the fastAI model
      """
      self.model.eval()
      device = CudaDeviceSingleton().get_device()
      model_loaded = self.model.to(device=device)
      # usually the last set of layers act as classifier, therefore we discard it    
      feature_extractor = model_loaded.features
      return feature_extractor
  
  def extract_features(self, images):
    """
    Extracts the features from a list of images
    :param images:
    :return: List[#images, #features]
    """
    features_bunch = self.feature_extractor(images)
    # pool of non-square window
    average_layer = nn.AvgPool2d((features_bunch.shape[2], features_bunch.shape[3]), stride=(1, 1))
    #averaging the features to lower the dimensionality in case is not wide resnet
    features_bunch = average_layer(features_bunch)
    features_bunch = features_bunch.view(-1, features_bunch.shape[1] * features_bunch.shape[2] * features_bunch.shape[3])

    return features_bunch