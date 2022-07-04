import torch
from MetaClasses.SingletonMeta import SingletonMeta

class CudaDeviceSingleton(metaclass=SingletonMeta):
    def __init__(self) -> None:
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def get_device(self) -> torch.device:
        return self.device