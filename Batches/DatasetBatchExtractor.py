import torch
from Batches.BalancedBatchSampler import BalancedBatchSampler
from CudaDeviceSingleton import CudaDeviceSingleton
from Batches.DatasetBatch import DatasetBatch
import torch.utils.data as tData

class DatasetBatchExtractor:
    def get_random_batch(dataset:tData.Dataset, batch_size:int) -> DatasetBatch:
        data_loader = tData.DataLoader(dataset = dataset, batch_size = batch_size, shuffle = True, drop_last = True)        
        return DatasetBatchExtractor.dataloader_to_batch(data_loader, batch_size)

    def get_balance_batch(dataset:tData.Dataset, batch_size:int) -> DatasetBatch: 
        sampler = BalancedBatchSampler(dataset)                                                                               
        data_loader = tData.DataLoader(dataset, batch_size=batch_size, sampler = sampler, drop_last = True)    
        return DatasetBatchExtractor.dataloader_to_batch(data_loader, batch_size)

    def get_mix_batch(dataset:tData.Dataset, ood_dataset:tData.Dataset, batch_size:int, ood_percentage:float) -> DatasetBatch:
        if ood_percentage == 0:
            return DatasetBatchExtractor.get_random_batch(dataset, batch_size)
        elif ood_percentage == 1:
            return DatasetBatchExtractor.get_random_batch(ood_dataset, batch_size)

        ood_size = round(batch_size * ood_percentage)
        batch = DatasetBatchExtractor.get_random_batch(dataset, batch_size - ood_size)
        ood_batch = DatasetBatchExtractor.get_random_batch(ood_dataset, ood_size)
        images = torch.cat((batch.images, ood_batch.images), 0)
        labels = torch.cat((batch.labels, ood_batch.labels), 0)

        return DatasetBatch(images, labels, batch_size)
        
    def dataloader_to_batch(dataloader:tData.DataLoader, batch_size:int) -> DatasetBatch:
        device = CudaDeviceSingleton().get_device()
        data_iterator = iter(dataloader)
        image, label = next(data_iterator)
        
        #move data to specific device
        images = image.to(device)
        labels = label.to(device)
        
        return DatasetBatch(images, labels, batch_size)