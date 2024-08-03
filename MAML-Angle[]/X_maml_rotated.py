#!/usr/bin/env python3
#
# Copyright (c) Facebook, Inc. and its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
This example shows how to use higher to do Model Agnostic Meta Learning (MAML)
for few-shot Omniglot classification.
For more details see the original MAML paper:
https://arxiv.org/abs/1703.03400

This code has been modified from Jackie Loong's PyTorch MAML implementation:
https://github.com/dragen1860/MAML-Pytorch/blob/master/omniglot_train.py

Our MAML++ fork and experiments are available at:
https://github.com/bamos/HowToTrainYourMAMLPytorch
"""

import argparse

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
import  torchvision.transforms as transforms
from model_utils import model_utils
from models import SimpleNN, ConvNet, ConvNet_Sans_Dropout

from X_load import MNISTNShot, RotatedMNIST


def main():

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--output_dir', type=str, help='Directory to save output files', default='Nouvelle_version/MAML/output/ConvNet/n_inner_iter=9_inner_opt_train_lr=0.05_meta_lr=5e-4'),
    argparser.add_argument('--n_way', type=int, help='n way', default=10) #par défaut 10
    argparser.add_argument('--k_spt', type=int, help='k shot for support set', default=10) #par défaut 10
    argparser.add_argument('--k_qry', type=int, help='k shot for query set', default=15) #par défaut 15
    argparser.add_argument('--num_classes', type=int, help='number of classes', default=20) #par défaut 20
    argparser.add_argument('--num_elements_per_class', type=int, help='number of elements per class', default=35) #par défaut 35. Doit être inférieur à 1000 car le test set contient 10k éléments
    argparser.add_argument('--perc_training_set', type=float, help='percentage of the training set', default=0.7) #par défaut 0.7
    argparser.add_argument('--task_num', type=int, help='meta batch size, namely task num', default=10) 
    argparser.add_argument('--epochs', type=int, help='number of epochs', default=300)  #par défaut 100
    argparser.add_argument('--meta_lr', type=float, help='learning rate of the meta optimizer', default=5e-4)  #par défaut 1e-3
    argparser.add_argument('--model_class', type=str, help='model class', default='ConvNet_Sans_Dropout')  #par défaut ConvNet   #A UTILISER 
    argparser.add_argument('--seed', type=int, help='random seed', default=1) #par défaut 1
    argparser.add_argument('--inner_opt_train_lr', type=float, help='inner optimization training learning rate', default=0.05) #par défaut 0.1
    argparser.add_argument('--inner_opt_test_lr', type=float, help='inner optimization training learning rate', default=0.1) #par défaut 0.1
    argparser.add_argument('--n_inner_iter', type=int, help='number of inner optimization steps', default=9) #par défaut 5

    args = argparser.parse_args()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    # Set up the Rotated MNIST loader.
    device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu")


    transform_pipeline = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),  # Ensure single channel
        transforms.Resize((28, 28)),  # Resize image
        transforms.ToTensor(),  # Convert to tensor and normalize to [0, 1]
        transforms.Normalize((0.1307,), (0.3081,))
        ])


    dataset=RotatedMNIST(root="data", train=True, transform=transform_pipeline, download=True, num_classes= args.num_classes, num_elements_per_class= args.num_elements_per_class)



#Vérifier le training set et le dataset ici

    db = MNISTNShot(dataset,  # Assuming training_data is loaded with rotated MNIST
    batchsz=args.task_num,
    n_way=args.n_way,
    k_shot=args.k_spt,
    k_query=args.k_qry,
    imgsz=28,  #UTILE car on fait déjà ça dans transform non ? # We pad images to 32x32 to match network requirements
    num_elements_per_class=args.num_elements_per_class, 
    perc_training_set=args.perc_training_set,  # 70% of the data is used for training
    device=device)

     
    # Create a vanilla PyTorch neural network that will be
    # automatically monkey-patched by higher later.
    # Before higher, models could *not* be created like this
    # and the parameters needed to be manually updated and copied
    # for the updates.
    net = ConvNet(num_classes=args.n_way).to(device)
    

    # We will use Adam to (meta-)optimize the initial parameters
    # to be adapted.
    meta_opt = optim.Adam(net.parameters(), lr=args.meta_lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(meta_opt, T_max=args.epochs)


    log = []

    for epoch in range(args.epochs):  #100 de base

        model_utils.train(db, net, device, meta_opt, scheduler, epoch, log, args.inner_opt_train_lr, args.n_inner_iter)
        model_utils.test(db, net, device, epoch, log, args.inner_opt_test_lr, args.n_inner_iter)
        model_utils.plot(log,args) 

        scheduler.step()
    
    # print("Unique values in true_labels:", np.unique(true_labels))
    # print("Unique values in predictions:", np.unique(predictions))
    
    # Sauvegarder les résultats
    model_utils.save_results(args, log)

    # cardinal_test_set =args.num_classes- int(args.perc_training_set * args.num_classes)
    # class_names = [str(i) for i in range(cardinal_test_set)]

    # model_utils.plot_confusion_matrix(true_labels, predictions, class_names, args.output_dir)
    model_utils.save_results(args, log)




# Won't need this after this PR is merged in:
# https://github.com/pytorch/pytorch/pull/22245
class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


if __name__ == '__main__':
    main()
