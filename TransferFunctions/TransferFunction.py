from abc import ABC, abstractmethod
from typing import List
from DatasetBatch import DatasetBatch

class TransferFunction(ABC):
    """
    Abstract delegate class to define transfer function
    """
    @abstractmethod
    def filter_batch(self, images_score:List[float]) -> List[bool]:
        """
        Return a bool for each images_score determining if it should be augmented or not
        """
        pass