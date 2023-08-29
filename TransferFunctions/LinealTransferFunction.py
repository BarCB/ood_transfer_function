from typing import List
from TransferFunctions.TransferFunction import TransferFunction

class LinealTransferFunction(TransferFunction):
    """
        Transfer function that assigns an augmentation probability based on how well the score is,
        the worst score will receive a probability of zero and as the score goes better the probability
        raises      
        
    """
    def filter_batch(self, images_score:List[float]) -> List[float]:
        augmentation_probabilities = []
        if len(images_score) == 1:
            augmentation_probabilities.append(1)
            return augmentation_probabilities

        max_score = max(images_score)        
        for score in images_score:
            probability = 100-(score*(100/max_score))
            augmentation_probabilities.append(probability/100)

        return augmentation_probabilities

