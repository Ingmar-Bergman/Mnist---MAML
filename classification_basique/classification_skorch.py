import argparse
import time
import typing

import pandas as pd
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('bmh')

import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor
import random
from sklearn.model_selection import GridSearchCV
from skorch import NeuralNetClassifier
import models

def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--epochs', type=int, help='Number of epochs for training', default=40)
    argparser.add_argument('--lr', type=float, help='Learning rate for optimizer', default=1e-4)
    argparser.add_argument('--seed', type=int, help='Random seed', default=1)
    argparser.add_argument('--num_classes', type=int, help='Number of classes in the dataset', default=30)
    argparser.add_argument('--test_split', type=float, help='Proportion of data to be used as test set', default=0.3)
    argparser.add_argument('--num_elements_per_class', type=int, help='Number of elements per class', default=70)
    argparser.add_argument('--batch_size', type=int, help='Size of the batch', default=32)

    args = argparser.parse_args()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    transform_pipeline = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
    ])

    db=RotatedMNIST(root="data", train=True, transform=transform_pipeline, download=True, num_classes= args.num_classes, num_elements_per_class= args.num_elements_per_class)

    unique_rotations = list(sorted(set(label for _, label in db.transformed_data)))
    rotation_to_label = {rotation: idx for idx, rotation in enumerate(unique_rotations)}

    #print informations about the labels    
    print(f"Unique rotations: {unique_rotations}") #Affiche les rotations uniques
    print(f"Rotation to label: {rotation_to_label}") #Affiche la correspondance entre les rotations et les labels


     # Rename the labels
    db.transformed_data = [(img, rotation_to_label[label]) for img, label in db.transformed_data]

    # Conversion en torch.float32
    db.transformed_data = [(transforms.ToTensor()(img).float(), label) for img, label in db.transformed_data]
    #rename the labels to be in the range of 0 to num_classes - 1
    
    #split the dataset into train and test

    random.shuffle(db.transformed_data)

    # Séparer les données en ensembles d'entraînement et de test
    split_idx = int(len(db.transformed_data) * (1 - args.test_split))
    train_data = db.transformed_data[:split_idx]
    test_data = db.transformed_data[split_idx:]

    # Créer les loaders de données pour l'entraînement et le test
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=32, shuffle=False)



    net = NeuralNetClassifier(
        SimpleNN,
        criterion=nn.CrossEntropyLoss,
        optimizer=optim.Adam,
        max_epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        module__num_classes=args.num_classes,  # Spécifier num_classes ici
        device=device
    )
    optimizer = optim.Adam(net.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()



    log = []
    for epoch in range(args.epochs):
        train(train_loader, net, device, optimizer, criterion, epoch, log)
        test(test_loader, net, device, epoch, log)
        plot(log)

def train(train_loader, net, device, optimizer, criterion, epoch, log):
    net.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (x, y) in enumerate(train_loader):
        x, y = x.to(device), y.to(device)

        #print y
        # print(f"Shape of y: {y.shape}")


        # Reshape x to (batch_size, 1, 28, 28)
        x = x.view(x.size(0), 1, 28, 28)

        # print(f"Shape of x: {x.shape}")
        optimizer.zero_grad()
        # print(f"Type of x: {x.dtype}")
        outputs = net(x)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += y.size(0)
        correct += predicted.eq(y).sum().item()

        if batch_idx % 100 == 0:
            print(f'[Epoch {epoch + 1}, Batch {batch_idx}] Loss: {loss.item():.4f} | Accuracy: {100. * correct / total:.2f}%')

    train_loss = total_loss / len(train_loader)
    train_acc = 100. * correct / total
    print(f'Train Epoch {epoch + 1}: Average Loss: {train_loss:.4f} | Accuracy: {train_acc:.2f}%')

    log.append({
        'epoch': epoch + 1,
        'loss': train_loss,
        'acc': train_acc,
        'mode': 'train',
        'time': time.time(),
    })

def test(test_loader, net, device, epoch, log):
    net.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(test_loader):
            x, y = x.to(device), y.to(device)

            # Reshape x to (batch_size, 1, 28, 28)
            x = x.view(x.size(0), 1, 28, 28)

            outputs = net(x)
            loss = F.cross_entropy(outputs, y)
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += y.size(0)
            correct += predicted.eq(y).sum().item()

    test_loss = total_loss / len(test_loader)
    test_acc = 100. * correct / total
    print(f'Test Epoch {epoch + 1}: Average Loss: {test_loss:.4f} | Accuracy: {test_acc:.2f}%')

    log.append({
        'epoch': epoch + 1,
        'loss': test_loss,
        'acc': test_acc,
        'mode': 'test',
        'time': time.time(),
    })

def plot(log):
    df = pd.DataFrame(log)
    fig, ax = plt.subplots(figsize=(6, 4))
    train_df = df[df['mode'] == 'train']
    test_df = df[df['mode'] == 'test']
    ax.plot(train_df['epoch'], train_df['acc'], label='Train')
    ax.plot(test_df['epoch'], test_df['acc'], label='Test')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    fig.legend(ncol=2, loc='lower right')
    fig.tight_layout()
    fname = 'classic_cnn_mnist.png'
    print(f'--- Plotting accuracy to {fname}')
    fig.savefig(fname)
    plt.close(fig)




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
        self.rotations = random.sample(range(361), nombre_rotations)

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




class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

if __name__ == '__main__':
    main()
