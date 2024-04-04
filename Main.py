import shutil
from Batches.DatasetBatch import DatasetBatch
from DatasetFactory import DatasetFactory
from DatasetsEnum import DatasetsEnum
from Batches.DatasetBatchExtractor import DatasetBatchExtractor
from OODScores.MahalanobisScore import MahalanobisScore
from torchvision.utils import save_image
from pathlib import Path
from OODScores.ScoreDelegate import ScoreDelegate
from TransferFunctions.PercentageTransferFunction import PercentageTransferFunction
from TransferFunctions.TransferFunction import TransferFunction
import AugmentationUtils as AT
from PIL import Image
from TransferFunctions.TransferFunctionEnum import TransferFunctionEnum
from TransferFunctions.TransferFunctionFactory import TransferFunctionFactory

def augmentate_images(augmentations_probabilities, batch:DatasetBatch, destination_folder:Path):
    for image_index in range(len(augmentations_probabilities)):
        probability = augmentations_probabilities[image_index]
        images = AT.augment_image(batch.getImages()[image_index], probability) 
    
        pil_image = Image.fromarray(images[1])
        image_fullname = Path(destination_folder, str(batch.labels[image_index].item()))
        pil_image.save(Path(image_fullname, f"{str(image_index)}_original.png"))

        was_augmented = images[0]
        if(was_augmented):
            pil_image = Image.fromarray(images[2])
            pil_image.save(Path(image_fullname, f"{str(image_index)}_augmented.png"))

def create_destination_folder(destination_path:Path):
    if destination_path.exists() and destination_path.is_dir():
        shutil.rmtree(destination_path)
    destination_path.mkdir(parents=True)

def create_experiment(batch_quantity:int, target_dataset, score:ScoreDelegate, 
                      transfer_function:TransferFunction, destination_folder:Path, batch_size_target):
    create_destination_folder(destination_folder)
    number_categories = len(target_dataset.class_to_idx)
    for current_batch in range(0, batch_quantity):
        print("Current batch: ", current_batch)
        target_batch = DatasetBatchExtractor.get_random_batch(target_dataset, batch_size_target)
        print("Unlabeled images shape(#images, channels, x, y): ", target_batch.images.shape)
        scores_for_batch = score.score_batch(target_batch)
        augmentation_probabilities = transfer_function.filter_batch(scores_for_batch)
        #probabilities per image
        batch_path = Path(destination_folder, "batch_" + str(current_batch))
        train_path = Path(batch_path, "train")

        for i in range(0, number_categories):
            Path(train_path, str(i)).mkdir(parents=True)

        augmentate_images(augmentation_probabilities, target_batch, train_path)

def generate_target_batches(weights_path:Path, batch_quantity:int, datasets_path:Path, destination_folder:Path):
    factory = DatasetFactory(datasets_path)  
    mahanobis_score = MahalanobisScore()
    mahanobis_score.load_weights(Path("featuresExtracted", weights_path))
    for batch_size_target in target_number_images:
        for target_dataset_name in target_datasets:
            target_dataset = factory.create_dataset(target_dataset_name)        
            for transfer_function_type in transfer_functions:
                
                experiment_path = Path(destination_folder, "target", weights_path + "_" + target_dataset_name.value + 
                                       "_" + str(batch_size_target) + "_" + transfer_function_type.value)

                transfer_function = TransferFunctionFactory.create_transfer_function(transfer_function_type)
                create_experiment(batch_quantity, target_dataset, mahanobis_score, transfer_function, experiment_path, batch_size_target)   

def save_image_batch(train_batch:DatasetBatch, batch_path:Path, number_categories:int):
    for i in range(number_categories):
        Path(batch_path, str(i)).mkdir(parents=True, exist_ok=True)

    for image_index in range(train_batch.size):
        save_image(train_batch.images[image_index], Path(batch_path, str(train_batch.labels[image_index].item()), str(image_index) + ".png"))

# Experiment factors ------------------------------------------
target_datasets = [DatasetsEnum.SVHN]
target_number_images = [100, 500, 1000, 4000]
transfer_functions = [TransferFunctionEnum.LinealFunction, TransferFunctionEnum.NoneFunction, TransferFunctionEnum.StepFunctionNegative, TransferFunctionEnum.StepFunctionPositive]
datasets_path = "C:\\Users\\Barnum\\Desktop\\datasets"
destination_folder = "C:\\Users\\Barnum\\Desktop\\experiments7"
# Experiment factors ------------------------------------------

def generate_tests():
    dataset_name = DatasetsEnum.SVHN
    factory = DatasetFactory(datasets_path)  
    dataset = factory.create_dataset(dataset_name) 
    target_batch = DatasetBatchExtractor.get_balance_batch(dataset, 1000)
    number_categories = len(dataset.class_to_idx)
    save_image_batch(target_batch, Path(destination_folder, "tests", dataset_name.value), number_categories)

def generate_targets():
    # Parameters ------------------------------------------
    batch_quantity = 10
    
    
    weights_path = "MNIST_4000"
    a = ["MNIST_400","MNIST_4000", "MNIST_40000"]
    # Parameters ------------------------------------------
    for b in a:
        generate_target_batches(b, batch_quantity, datasets_path, destination_folder) 

def generate_sources():
    # Parameters ------------------------------------------
    source_batch_size = 40000  #MNIST has 42k images but for hardware capacity 25000 is used
    datasets_path = "C:\\Users\\Barnum\\Desktop\\datasets"
    destination_folder = "C:\\Users\\Barnum\\Desktop\\experiments7"
    factory = DatasetFactory(datasets_path)
    source_dataset = factory.create_dataset(DatasetsEnum.MNIST)
    source_batch = DatasetBatchExtractor.get_random_batch(source_dataset, source_batch_size)
    mahanobis_score = MahalanobisScore()
    mahanobis_score.create_weights(source_batch)
    mahanobis_score.save_weights(Path("featuresExtracted", "MNIST_" + str(source_batch_size)))

    number_categories = len(source_dataset.class_to_idx)
    save_image_batch(source_batch, Path(destination_folder, "sources", DatasetsEnum.MNIST.value + str(source_batch_size)), number_categories)

if __name__ == "__main__":
   generate_tests()