# utils.py

import time
import pandas as pd
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('bmh')

import torch
import torch.nn.functional as F
import os
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold
import seaborn as sns
import numpy as np

from torch.utils.data import DataLoader, TensorDataset

class model_utils:
    @staticmethod
    def train(train_loader, net, device, optimizer, criterion, epoch, log):
        net.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)

            # Reshape x to (batch_size, 1, 28, 28)
            x = x.view(x.size(0), 1, 28, 28)

            optimizer.zero_grad()
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

    @staticmethod
    def test(test_loader, net, device, epoch, log):
        net.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        true_labels = []
        predicted_labels = []

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

                # Collect true and predicted labels
                true_labels.extend(y.cpu().numpy())
                predicted_labels.extend(predicted.cpu().numpy())

        test_loss = total_loss / len(test_loader)
        test_acc = 100. * correct / total
        print(f'Test Epoch {epoch + 1}: Average Loss: {test_loss:.4f} | Accuracy: {test_acc:.2f}%')

        log.append({
            'epoch': epoch + 1,
            'loss': test_loss,
            'acc': test_acc,
            'mode': 'test',
            'time': time.time(),
            'true_labels': true_labels,
            'predicted_labels': predicted_labels
        })

    
    @staticmethod
    def cross_validation(model_class, args, dataset, num_folds=3, epochs=10, lr=0.001, batch_size=64, output_dir='output', device='cuda'): #model_class est la classe du modèle


        images = [item[0] for item in dataset.transformed_data]
        labels = [item[1] for item in dataset.transformed_data]

        # Convert to PyTorch tensors
        image_tensor = torch.stack(images)  # Assuming images are already tensors
        label_tensor = torch.tensor(labels)

        #print image_tensor shape
        print(f"image_tensor shape: {image_tensor.shape}")

        # Create a TensorDataset
        dataset = TensorDataset(image_tensor, label_tensor)


        # Initialisation de la cross-validation
        skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=0)
        fold_accuracies = []
        fold_scores = []  # List to store scores for each fold

        for fold, (train_indices, val_indices) in enumerate(skf.split(dataset.tensors[0], dataset.tensors[1])):
            print(f'Fold {fold + 1}/{num_folds}')

            # Création des loaders de données pour l'entraînement et la validation
            train_subset = torch.utils.data.Subset(dataset, train_indices)
            val_subset = torch.utils.data.Subset(dataset, val_indices)
            train_loader = torch.utils.data.DataLoader(train_subset, batch_size=batch_size, shuffle=True)
            val_loader = torch.utils.data.DataLoader(val_subset, batch_size=batch_size, shuffle=False)


            model = model_class(args.num_classes).to(device)

            # Création du modèle, critère de perte et optimiseur
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            criterion = torch.nn.CrossEntropyLoss()

            # Log pour enregistrer les métriques d'entraînement et de validation
            log = []

            # for batch_idx, (x, y) in enumerate(train_loader):
            #     print(f"Batch {batch_idx}: Input shape = {x.shape}, Target shape = {y.shape}")


            # Entraînement et validation du modèle pour chaque fold
            for epoch in range(epochs):
                model_utils.train(train_loader, model, device, optimizer, criterion, epoch, log)
                model_utils.test(val_loader, model, device, epoch, log)

            # Collecte de l'accuracy de validation pour ce fold
            val_acc = [entry['acc'] for entry in log if entry['mode'] == 'test'][-1]  # Récupère la dernière entrée de validation
            fold_accuracies.append(val_acc)
            fold_scores.append(log[-1]['acc'])  # Ajoute le score de validation du dernier epoch




        # Calcul de la moyenne et de la variance des accuracies de validation
        mean_accuracy = np.mean(fold_accuracies)
        variance_accuracy = np.var(fold_accuracies)

        print(f'Mean Accuracy across {num_folds} folds: {mean_accuracy:.2f}%')
        print(f'Variance of Accuracy across {num_folds} folds: {variance_accuracy:.2f}')

        # Sauvegarde des résultats
        os.makedirs(output_dir, exist_ok=True)
        results_file = os.path.join(output_dir, 'cross_validation_results_ConvNet.txt')
        #do os path join with at the end of the file the class model
        results_file1=os.path.join(output_dir, f'cross_validation_results_{model_class}.txt')
        with open(results_file, 'w') as f:
            f.write(f'Arguments:\n')
            f.write(f' - Class of the model: {model_class}\n')
            f.write(f' - Number of elemnts per class: {args.num_elements_per_class}\n')
            f.write(f' - Number of folds: {num_folds}\n')
            f.write(f' - Number of epochs: {epochs}\n')
            f.write(f' - Learning rate: {lr}\n')
            f.write(f' - Batch size: {batch_size}\n')
            f.write(f' - Output directory: {output_dir}\n\n')


            # f.write(f'Mean Accuracy across {num_folds} folds: {mean_accuracy:.2f}%\n')
            # f.write(f'Variance of Accuracy across {num_folds} folds: {variance_accuracy:.2f}\n')
            f.write(f'Mean Accuracy across {num_folds} folds: {mean_accuracy:.2f}%\n')
            f.write(f'Variance of Accuracy across {num_folds} folds: {variance_accuracy:.2f}\n\n')
            f.write('Individual Fold Scores:\n')
            for fold, score in enumerate(fold_scores):
                f.write(f'Fold {fold + 1} Validation Accuracy: {score:.2f}%\n')
                f.write(f'Fold {fold + 1} Results File: fold_{fold + 1}_results_ConvNet.txt\n')

        print(f'Cross-validation results saved to {results_file}')

    @staticmethod
    def plot(log,output_dir='output'):
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
        # fname = 'classic_cnn_mnist.png'
        # print(f'--- Plotting accuracy to {fname}')
        # fig.savefig(fname)
        # plt.close(fig)

        # Create the directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Construct the filename based on args
        fname = 'classic_cnn_mnist.png'
        fname = os.path.join(output_dir, fname)
        
        print(f'--- Plotting accuracy to {fname}')
        fig.savefig(fname)
        plt.close(fig)

    
    @staticmethod
    def save_results(args, log, output_dir='output'):
        os.makedirs(output_dir, exist_ok=True)
        file_path = os.path.join(output_dir, 'results_CNN.txt')

        with open(file_path, 'w') as f:
            f.write(f'Hyperparameters:\n')
            for arg in vars(args):
                f.write(f'{arg}: {getattr(args, arg)}\n')

            f.write('\nAccuracy Results:\n')

            for i, entry in enumerate(log):
                if i % 2 == 0:
                    f.write(f"Epoch {entry['epoch']}: Train Accuracy = {entry['acc']:.2f}%, ")
                else:
                    f.write(f"Test Accuracy = {entry['acc']:.2f}%\n")

                

        print(f'--- Saved results to {file_path}')

    @staticmethod
    def plot_confusion_matrix(y_true, y_pred, rotation_to_label, output_dir='output'):
        os.makedirs(output_dir, exist_ok=True)
        file_path = os.path.join(output_dir, 'confusion_matrix.png')

        # Inverse the rotation_to_label dictionary to map indices back to rotations
        label_to_rotation = {v: k for k, v in rotation_to_label.items()}

        # Convertir les étiquettes numériques en rotations correspondantes
        y_true_rotations = [label_to_rotation[label] for label in y_true]
        y_pred_rotations = [label_to_rotation[label] for label in y_pred]

        # Compute confusion matrix
        cm = confusion_matrix(y_true_rotations, y_pred_rotations)
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]  # Normalize

        # Plot confusion matrix
        plt.figure(figsize=(20, 14))
        sns.set(font_scale=0.9)
        sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues", xticklabels=sorted(label_to_rotation.values()), yticklabels=sorted(label_to_rotation.values()))
        plt.xlabel('Predicted labels')
        plt.ylabel('True labels')
        plt.title('Confusion Matrix')
        plt.savefig(file_path)
        plt.close()

        print(f'--- Saved confusion matrix to {file_path}')

    @staticmethod
    def get_accuracy_label(label, y_true, y_pred): # label between 0 and num_classes - 1
        # Calculate accuracy for label 0
        correct_0 = sum(1 for t, p in zip(y_true, y_pred) if t == label and p == label)
        total_0 = sum(1 for t in y_true if t == label)
        accuracy_0 = 100. * correct_0 / total_0 if total_0 > 0 else 0.0
        print(f"Accuracy for label {label}: {accuracy_0:.2f}%")

