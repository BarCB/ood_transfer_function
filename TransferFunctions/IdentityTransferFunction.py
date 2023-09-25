from typing import List
from TransferFunctions.TransferFunction import TransferFunction

class IdentityTransferFunction(TransferFunction):
    """
        Transfer function that marks all the images as candidates to be augmented
    """
    def filter_batch(self, images_score:List[float]) -> List[float]:
        images_to_augmentate = [1] * len(images_score)
        return images_to_augmentate

