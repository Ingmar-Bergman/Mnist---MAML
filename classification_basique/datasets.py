import pandas as pd
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('bmh')

import torch
from torch import nn
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor
import random



class RotatedMNIST(Dataset):
    def __init__(self, root, train=True, transform=None, download=False, num_classes=30, num_elements_per_class=200):
        train_data = datasets.MNIST(root=root, train=True, download=download)
        test_data = datasets.MNIST(root=root, train=False, download=download)
        
        # Concatenate train and test datasets
        self.original_datasets = train_data + test_data

        self.transform = transform

        # Paramètres par défaut
        self.num_elements_per_class = num_elements_per_class
        self.nombre_rotations = num_classes

        # Générer les données transformées
        self.transformed_data = self.generate_transformed_data(self.num_elements_per_class, self.nombre_rotations)

        self.normalize_data()

    
    def generate_transformed_data(self, num_elements_per_class, nombre_rotations):
        # Liste pour stocker les datasets transformés
        transformed_datasets = []
        self.rotations = np.linspace(0, 360, nombre_rotations, endpoint=False).astype(int)

        count = 0
        temp_images = []
        rotation_index = 0  # Pour suivre l'index des rotations

        for i in range(len(self.original_datasets)):
            if rotation_index >= len(self.rotations):
                break  # Arrêter l'itération si toutes les rotations ont été utilisées

            image, _ = self.original_datasets[i]  # Ignorer le label original
            temp_images.append(image)
            count += 1

            # Une fois que nous avons le nombre spécifié d'images, appliquez une rotation unique et mettez à jour les labels
            if count % num_elements_per_class == 0:
                rotation = self.rotations[rotation_index]
                rotation_index += 1
                rotated_images = [img.rotate(rotation) for img in temp_images]
                rotated_images = [np.expand_dims(np.array(img), axis=0) for img in rotated_images] 
                transformed_datasets.append([(img, rotation) for img in rotated_images])
                temp_images = []

                # Vérifier si toutes les rotations ont été utilisées
                if rotation_index >= len(self.rotations):
                    break

        # Aplatir la liste de listes en une seule liste
        transformed_data = [item for sublist in transformed_datasets for item in sublist]
        
        return transformed_data

        
    def normalize_data(self):
        """
        Normalize the data to have a mean of 0 and a std of 1
        """
        x_data = np.array([img for img, _ in self.transformed_data])
        self.mean = np.mean(x_data)
        self.std = np.std(x_data)
        self.max = np.max(x_data)
        self.min = np.min(x_data)

        print("before norm:", "mean", self.mean, "max", self.max, "min", self.min, "std", self.std)

        x_data = (x_data - self.mean) / self.std

        self.transformed_data = [(x_data[i], label) for i, (_, label) in enumerate(self.transformed_data)]

        self.mean = np.mean(x_data)
        self.std = np.std(x_data)
        self.max = np.max(x_data)
        self.min = np.min(x_data)

        print("after norm:", "mean", self.mean, "max", self.max, "min", self.min, "std", self.std)


    def __len__(self):
        return len(self.transformed_data)

    def __getitem__(self, idx):
        image, label = self.transformed_data[idx]
        if self.transform:
            image = self.transform(image)
        return image, label