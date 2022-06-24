import torchvision.datasets
import numpy
from typing import Any, Callable, cast, Dict, List, Optional, Tuple

class CustomDataset(torchvision.datasets.ImageFolder):
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            augmented = self.transform(image=numpy.array(sample)) 
            sample = augmented['image']
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target