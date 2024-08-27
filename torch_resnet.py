from pathlib import Path
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from torchvision.models import resnet50
from torch.utils.data import DataLoader
import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "caching_allocator"

# Define transforms to preprocess the data
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize images to fit ResNet50 input size
    transforms.ToTensor(),  # Convert PIL images to tensors
])

CATEGORIES_NUMBER = 2
BATCH_QUANTITY = 30

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def create_model(freeze_layers=False):
    # Define ResNet50 model
    model = resnet50(weights=None)  # Load pre-trained ResNet50 weights
    if freeze_layers:
        for param in model.parameters():
            param.requires_grad = False

    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, CATEGORIES_NUMBER)  # Change the last fully connected layer to output 10 classes for MNIST
    model = model.to(device)
    return model

def train_model(model, train_dataset, learning_rate = 0.001, momentum=0.9):
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)

    # Train the model
    model.train()
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    for epoch in range(10):  # Example of 5 epochs, you can adjust as needed
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 5 == 4:  # Print every 100 mini-batches
                print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / 100))
                running_loss = 0.0

    print('Finished Training')

def evaluate_model(model, test_dataset):
    # Optionally, evaluate the model
    model.eval()
    correct = 0
    total = 0
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    with torch.no_grad():
        for data in test_loader:
            inputs, labels = data[0].to(device), data[1].to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    print('Accuracy of the network on the test images: %f %%' % (accuracy))
    return accuracy

def delete_empty_folders(path):
    for root, dirs, files in os.walk(path, topdown=False):
        for directory in dirs:
            current_dir = os.path.join(root, directory)
            if not os.listdir(current_dir):
                print(f"Deleting empty directory: {current_dir}")
                os.rmdir(current_dir)

def list_folders(path):
    if not os.path.exists(path):
        print("The specified path does not exist.")
        return []

    entries = os.listdir(path)
    folders = [entry for entry in entries if os.path.isdir(os.path.join(path, entry))]

    print(folders)

def domain_adaptation_training(location_path, sources):
    for source in sources:
        exp_conditions = source.split("_")

        test_path = Path(location_path, "tests", exp_conditions[0])
        test_dataset = torchvision.datasets.ImageFolder(root=test_path, transform=transform)
        for i in range(BATCH_QUANTITY):
            print("Batch: ", i)
            model = create_model()
    
            source_path = Path(location_path, "sources", source, "batch_" + str(i), "train") 
            source_dataset = torchvision.datasets.ImageFolder(root=source_path, transform=transform)
            train_model(model, source_dataset, learning_rate=0.01, momentum=0)

            target_path = Path(location_path, "targets", exp_conditions[0] + "_" + exp_conditions[1]) 
            target_dataset = torchvision.datasets.ImageFolder(root=target_path, transform=transform)
            train_model(model, target_dataset)
            accuracy = evaluate_model(model, test_dataset)

            file1 = open("results.txt", "a")
            file1.write(f"{exp_conditions[2]}\t'{exp_conditions[3]}'\t{exp_conditions[0]}\t'{exp_conditions[1]}'\t{exp_conditions[4]}\t{accuracy}\t{exp_conditions[5]}\t'{exp_conditions[6]}'")
            file1.write("\n")
            file1.close()

def main():
    print("Is CUDA available?", torch.cuda.is_available())
    location_path = f"C:\\Users\\Barnum\\Desktop\\experiments13\\"

    sources = ['CR_130_CR_100_IdentityFunctionPositive_CatsVsDogs_0.25', 'CR_130_CR_100_IdentityFunctionPositive_CatsVsDogs_0.5', 'CR_130_CR_100_IdentityFunctionPositive_China_0.25', 'CR_130_CR_100_IdentityFunctionPositive_China_0.5', 'CR_130_CR_100_IdentityFunctionPositive_CRSnP_0.25', 'CR_130_CR_100_IdentityFunctionPositive_CRSnP_0.5', 'CR_130_CR_100_LinealFunction_CatsVsDogs_0.25', 'CR_130_CR_100_LinealFunction_CatsVsDogs_0.5', 'CR_130_CR_100_LinealFunction_China_0.25', 'CR_130_CR_100_LinealFunction_China_0.5', 'CR_130_CR_100_LinealFunction_CRSnP_0.25', 'CR_130_CR_100_LinealFunction_CRSnP_0.5', 'CR_130_CR_100_NoneFunction_CatsVsDogs_0.25', 'CR_130_CR_100_NoneFunction_CatsVsDogs_0.5', 'CR_130_CR_100_NoneFunction_China_0.25', 'CR_130_CR_100_NoneFunction_China_0.5', 'CR_130_CR_100_NoneFunction_CRSnP_0.25', 'CR_130_CR_100_NoneFunction_CRSnP_0.5', 'CR_130_CR_100_StepFunctionNegative_CatsVsDogs_0.25', 'CR_130_CR_100_StepFunctionNegative_CatsVsDogs_0.5', 'CR_130_CR_100_StepFunctionNegative_China_0.25', 'CR_130_CR_100_StepFunctionNegative_China_0.5', 'CR_130_CR_100_StepFunctionNegative_CRSnP_0.25', 'CR_130_CR_100_StepFunctionNegative_CRSnP_0.5', 'CR_130_CR_100_StepFunctionPositive_CatsVsDogs_0.25', 'CR_130_CR_100_StepFunctionPositive_CatsVsDogs_0.5', 'CR_130_CR_100_StepFunctionPositive_China_0.25', 'CR_130_CR_100_StepFunctionPositive_China_0.5', 'CR_130_CR_100_StepFunctionPositive_CRSnP_0.25', 'CR_130_CR_100_StepFunctionPositive_CRSnP_0.5']
    
    domain_adaptation_training(location_path, sources)

    list_folders(Path(location_path, "sources"))

if __name__ == "__main__":
    main()
