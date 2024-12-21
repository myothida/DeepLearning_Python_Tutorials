import os
import pandas as pd
import json
import requests

from ucimlrepo import fetch_ucirepo 
from sklearn.datasets import fetch_openml, load_iris
from torchvision import datasets, transforms


class DataLoader:
    def __init__(self):
        # Root directory for saving datasets
        self.data_dir = './data'
        self.ensure_directory_exists(self.data_dir)
        
        self.check_dataid = {
            "auto_mpg": self.load_auto_mpg,            
            "mnist": self.load_mnist,
            "car": self.load_car
        }
        
        self.loaded_datasets = {}
        
    def ensure_directory_exists(self, directory):
        if not os.path.exists(directory):
            print(f"Creating directory: {directory}")
            os.makedirs(directory, exist_ok=True)


    def load_auto_mpg(self):
        """Load the Auto MPG dataset from UCIMLrepo, if not already downloaded."""
        dataset_path = os.path.join(self.data_dir, 'auto_mpg.csv')
        if not os.path.exists(dataset_path):
            print("Downloading Auto MPG dataset...")
            auto_mpg = fetch_ucirepo(id=9) 
            X = auto_mpg.data.features 
            y = auto_mpg.data.targets 
            dataset = pd.concat([X,y], axis = 1)
            dataset.to_csv(dataset_path, index=False)
        else:
            print("Loading Auto MPG dataset from local storage...")
            dataset = pd.read_csv(dataset_path)
        return dataset


    def load_mnist(self):
        """Load the MNIST dataset from torchvision, if not already downloaded."""
        mnist_path = os.path.join(self.data_dir, 'MNIST')
        if not os.path.exists(mnist_path):
            print("Downloading MNIST dataset...")
            transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,), (0.5,))])
            mnist_trainset = datasets.MNIST(root=self.data_dir, train=True, download=True, transform=transform)
            mnist_testset = datasets.MNIST(root=self.data_dir, train=False, download=True, transform=transform)
            print("MNIST dataset downloaded.")
        else:
            print("MNIST dataset already downloaded")
            transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,), (0.5,))])
            mnist_trainset = datasets.MNIST(root=self.data_dir, train=True, download=False, transform=transform)
            mnist_testset = datasets.MNIST(root=self.data_dir, train=False, download=False,transform=transform)
        return mnist_trainset, mnist_testset

    def load_car(self):
        """Load the car images."""
        data_dir =  os.path.join(self.data_dir, 'car_truck')
        train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])

        test_transforms = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])

        # The ImageFolder class is used to load images from a directory structure where the folder names represent class
        train_data = datasets.ImageFolder(data_dir + '/train', transform=train_transforms)
        test_data = datasets.ImageFolder(data_dir + '/test', transform=test_transforms)

        return train_data, test_data

    def get_dataset(self, dataset_id):
        """Retrieve a dataset based on its name, loading it if necessary."""
        if dataset_id not in self.check_dataid:
            raise ValueError(f"Dataset ID '{dataset_id}' not found. Please check the available datasets.")
        
        if dataset_id not in self.loaded_datasets:
            print(f"Loading dataset: {dataset_id}...")
            self.loaded_datasets[dataset_id] = self.check_dataid[dataset_id]()
        return self.loaded_datasets[dataset_id]
    
    def imagenet1000_cls_id_label(self):
        dataset_path = os.path.join(self.data_dir, 'imagenet_class_labels.json')
        if not os.path.exists(dataset_path):
            # URL for ImageNet class labels
            labels_url = "https://storage.googleapis.com/download.tensorflow.org/data/imagenet_class_index.json"
            response = requests.get(labels_url)
            if response.status_code == 200:
                class_idx_to_label = response.json()                  
                with open(dataset_path, "w") as json_file:
                    json.dump(class_idx_to_label, json_file, indent=4)  
                print(f"JSON file saved successfully as '{dataset_path}'!")
            else:
                print(f"Failed to fetch JSON. HTTP Status Code: {response.status_code}")
        else:
            with open(dataset_path, "r") as imagenet_class_file:
                class_idx_to_label  = json.load(imagenet_class_file)
            return class_idx_to_label


    
    def list_datasets(self):
        """List all available dataset IDs."""
        return list(self.check_dataid.keys())
