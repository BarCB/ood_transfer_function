from typing import List
from TransferFunctions.TransferFunction import TransferFunction

class IdentityTransferFunction(TransferFunction):
    """
        Transfer function that marks all the images as candidates to be augmented
    """
    def __init__(self, should_augment:bool) -> None:
        self.should_augment = should_augment
        super().__init__()

    def filter_batch(self, images_score:List[float]) -> List[float]:
        images_to_augmentate = [self.should_augment] * len(images_score)
        return images_to_augmentate

