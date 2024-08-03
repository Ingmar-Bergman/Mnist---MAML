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
from torchvision import transforms
from model_utils import model_utils
from datasets import RotatedMNIST
import random
from models import SimpleNN, ConvNet, ConvNet_Sans_Dropout
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import  TensorDataset


def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--output_dir', type=str, help='Directory to save output files', default='classification_basique/output/test_test')
    argparser.add_argument('--epochs', type=int, help='Number of epochs for training', default=120)
    argparser.add_argument('--lr', type=float, help='Learning rate for optimizer', default=1e-3)
    argparser.add_argument('--seed', type=int, help='Random seed', default=1)
    argparser.add_argument('--num_classes', type=int, help='Number of classes in the dataset', default=25)
    argparser.add_argument('--test_split', type=float, help='Proportion of data to be used as test set', default=0.3)
    argparser.add_argument('--num_elements_per_class', type=int, help='Number of elements per class', default=20)
    argparser.add_argument('--batch_size', type=int, help='Size of the batch', default=32)
    argparser.add_argument('--scheduler', type=str, help='Scheduler type: none, cosine, or step', default='none')

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
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data, args.batch_size, shuffle=False)


    # net = nn.Sequential(
    #     nn.Conv2d(1, 64, 3),
    #     nn.BatchNorm2d(64, momentum=1, affine=True),
    #     nn.ReLU(inplace=True),
    #     nn.MaxPool2d(2, 2),
    #     nn.Conv2d(64, 64, 3),
    #     nn.BatchNorm2d(64, momentum=1, affine=True),
    #     nn.ReLU(inplace=True),
    #     nn.MaxPool2d(2, 2),
    #     nn.Conv2d(64, 64, 3),
    #     nn.BatchNorm2d(64, momentum=1, affine=True),
    #     nn.ReLU(inplace=True),
    #     nn.MaxPool2d(2, 2),
    #     Flatten(),
    #     nn.Linear(64, args.num_classes)  # Nombre de classes spécifié par l'utilisateur
    # ).to(device)

    net = ConvNet(num_classes=args.num_classes).to(device)

    # # Assuming db.transformed_data is a list of (image_tensor, label) tuples
    # images = [item[0] for item in db.transformed_data]
    # labels = [item[1] for item in db.transformed_data]

    # # Convert to PyTorch tensors
    # image_tensor = torch.stack(images)  # Assuming images are already tensors
    # label_tensor = torch.tensor(labels)

    # #print image_tensor shape
    # print(f"image_tensor shape: {image_tensor.shape}")

    # # Create a TensorDataset
    # dataset = TensorDataset(image_tensor, label_tensor)
        

    # model_utils.cross_validation(ConvNet, args, db, num_folds=3, epochs=args.epochs, lr=args.lr, batch_size=args.batch_size, output_dir=args.output_dir)

    optimizer = optim.Adam(net.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()


    scheduler = None
    if args.scheduler == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    elif args.scheduler == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)


    log = []
    for epoch in range(args.epochs):
        model_utils.train(train_loader, net, device, optimizer, criterion, epoch, log)
        model_utils.test(test_loader, net, device, epoch, log)
        model_utils.plot(log, output_dir=args.output_dir)
        if scheduler is not None:
            scheduler.step()

    
    
 # Après les tests de chaque époque dans main()
    y_true = []
    y_pred = []

    for entry in log:
        if entry['mode'] == 'test':
            y_true.extend(entry['true_labels'])
            y_pred.extend(entry['predicted_labels'])

    # Générer et sauvegarder la matrice de confusion
    model_utils.plot_confusion_matrix(y_true, y_pred, rotation_to_label, output_dir=args.output_dir)

    # Sauvegarder les résultats
    model_utils.save_results(args, log, args.output_dir)

    # Calculate accuracy for label 0
    model_utils.get_accuracy_label(0, y_true, y_pred)



class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

if __name__ == '__main__':
    main()
