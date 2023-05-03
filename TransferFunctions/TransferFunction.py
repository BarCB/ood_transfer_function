from abc import ABC, abstractmethod
from typing import List

class TransferFunction(ABC):
    """
    Abstract delegate class to define transfer function
    """
    @abstractmethod
    def filter_batch(self, images_score:List[float]) -> List[float]:
        """
        Return the augmentation probability for each images_score
        """
        pass