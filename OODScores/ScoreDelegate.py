from abc import ABC, abstractmethod
from typing import List
from Batches.DatasetBatch import DatasetBatch

class ScoreDelegate(ABC):
    """
    Abstract delegate class to define out of distribution score
    """
    @abstractmethod
    def score_batch(self, labeled_batch:DatasetBatch, unlabeled_batch:DatasetBatch) -> List[int]:
        """
        Return a score for each image in unlabeled_batch determining how close distribution is versus labeled_batch
        """
        pass