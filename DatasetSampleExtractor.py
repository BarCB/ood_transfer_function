import torch
from BalancedBatchSampler import BalancedBatchSampler
from CudaDeviceSingleton import CudaDeviceSingleton
from DatasetBatch import DatasetBatch

class DatasetSampleExtractor:
    def get_random_batch(dataset, batch_size) -> DatasetBatch:
        data_loader = torch.utils.data.DataLoader(dataset = dataset, batch_size = batch_size, shuffle = True, drop_last = True)        
        return DatasetSampleExtractor.dataloader_to_batch(data_loader, batch_size)

    def get_balance_batch(dataset:torch.utils.data.Dataset, batch_size:int) -> DatasetBatch: 
        sampler = BalancedBatchSampler(dataset)                                                                               
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler = sampler, drop_last = True)    
        return DatasetSampleExtractor.dataloader_to_batch(data_loader)

    def dataloader_to_batch(dataloader:torch.utils.data.DataLoader, batch_size:int) -> DatasetBatch:
        device = CudaDeviceSingleton().get_device()
        for image, label in dataloader:
            #move data to specific device
            images = image.to(device)
            labels = label.to(device)
        
        return DatasetBatch(images, labels, batch_size)