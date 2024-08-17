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
import time
import typing

import pandas as pd
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('bmh')
import os 
import torch
torch.cuda.empty_cache()

from torch import nn
import torch.nn.functional as F
import torch.optim as optim
import  torchvision.transforms as transforms

import higher
from model_utils import model_utils

from support.Load_Rotated_Equivariance_MNIST import MNISTNShot, RotatedMNIST
from layers import ShareConv2d
from inner_optimizers import InnerOptBuilder
from models import ConvNet, ConvNet_Sans_Dropout, ConvNet_with_Dropout


def main(args):


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

    #A VOIR SI JE GARDE LE TRUC DU BAS OU NON
    #transform grey level

    transform_pipeline = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),  # Ensure single channel
        transforms.Resize((28, 28)),  # Resize image
        transforms.ToTensor(),  # Convert to tensor and normalize to [0, 1]
        transforms.Normalize((0.1307,), (0.3081,))
        ])


    dataset=RotatedMNIST(root="data", train=True, transform=transform_pipeline, download=True, num_classes= args.num_classes, num_elements_per_class= args.num_elements_per_class)

    print(device)
#Vérifier le training set et le dataset ici

    db = MNISTNShot(dataset,  # Assuming training_data is loaded with rotated MNIST
    batchsz=args.task_num,
    n_way=args.n_way,
    k_shot=args.k_spt,
    k_query=args.k_qry,
    imgsz=28,  #UTILE car on fait déjà ça dans transform non ? # We pad images to 32x32 to match network requirements
    num_elements_per_class=args.num_elements_per_class,
    device=device)

    # Create a vanilla PyTorch neural network that will be
    # automatically monkey-patched by higher later.
    # Before higher, models could *not* be created like this
    # and the parameters needed to be manually updated and copied
    # for the updates.

    


    net = args.model_class(args.n_way).to(device)

    # We will use Adam to (meta-)optimize the initial parameters
    # to be adapted.
    # meta_opt = optim.Adam(net.parameters(), lr=1e-3)

    #MSR 
    inner_opt_builder = InnerOptBuilder(net, device, opt_name='maml_adam', init_lr=args.init_inner_lr, init_mode='learned', lr_mode='per_layer')
    meta_opt = optim.Adam(inner_opt_builder.metaparams.values(), lr=args.outer_lr) #prendre sgd pour le test
    inner_opt_builder_train = InnerOptBuilder(net, device, opt_name='maml', init_lr=args.inner_opt_builder_train_lr, init_mode='learned', lr_mode='per_layer')
    inner_opt_builder_test = InnerOptBuilder(net, device, opt_name='maml', init_lr=args.inner_opt_builder_test_lr, init_mode='learned', lr_mode='per_layer')
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(meta_opt, T_max=args.num_epochs)

    log = []
    for epoch in range(args.num_epochs):  #100 de base
        model_utils.train(db, net, device, inner_opt_builder_train, meta_opt, scheduler, epoch, log, args)
        
        #if epoch is the last epoch 
        if epoch == args.num_epochs - 1:
            all_true_labels, all_pred_labels= model_utils.test(db, net, device, inner_opt_builder_test, epoch, log, args)
        else : 
            model_utils.test(db, net, device, inner_opt_builder_test, epoch, log, args)
        model_utils.plot(log, args)

    model_utils.save_results(args, log)
    model_utils.compute_and_save_confusion_matrix(all_true_labels, all_pred_labels, args.output_dir)




if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--output_dir', type=str, help='Directory to save output files', default='Nouvelle_version/MSR/output/ConvNet'),
    argparser.add_argument('--n_way', type=int, help='n way', default=10) #par défaut 10
    argparser.add_argument('--k_spt', type=int, help='k shot for support set', default=10) #par défaut 10
    argparser.add_argument('--k_qry', type=int, help='k shot for query set', default=15) #par défaut 15
    argparser.add_argument('--num_elements_per_class', type=int, help='number of elements per class', default=35) #par défaut 35
    argparser.add_argument('--task_num', type=int, help='meta batch size, namely task num', default=32) #DE BASE CEST A 32
    argparser.add_argument('--num_classes', type=int, help='number of classes', default=20) #par défaut 20
    argparser.add_argument('--model_class', type=str, help='model class', default='ConvNet')
    argparser.add_argument('--seed', type=int, help='random seed', default=1) #par défaut 1
    argparser.add_argument('--init_inner_lr', type=float, help='initial inner learning rate', default=0.1) #par défaut 0.1
    argparser.add_argument('--inner_opt_builder_train_lr', type=float, help='inner optimization training learning rate', default=0.1) #par défaut 0.1
    argparser.add_argument('--inner_opt_builder_test_lr', type=float, help='inner optimization test learning rate', default=0.1) #par défaut 0.1
    argparser.add_argument('--outer_lr', type=float, help='outer learning rate', default=1e-3) #par défaut 1e-3
    argparser.add_argument('--num_inner_steps', type=int, help='number of inner loop steps', default=5) #par défaut 5
    argparser.add_argument('--num_epochs', type=int, help='number of epochs', default=50) #par défaut 50
    args = argparser.parse_args()

    # init_inner_lr_values = [0.1,0.001,0.0001]
    # inner_opt_builder_train_lr_values = [0.1,0.01]
    # inner_opt_builder_test_lr_values = [0.1,0.01]
    # outer_lr_values = [1e-3,1e-4]
    # num_inner_steps_values = [7]
    # num_epochs_values = [70]

    init_inner_lr_values = [0.0001]
    inner_opt_builder_train_lr_values = [0.1]
    inner_opt_builder_test_lr_values = [0.1]
    outer_lr_values = [1e-3]
    num_inner_steps_values = [5]
    num_epochs_values = [70]



    for init_inner_lr in init_inner_lr_values:
        for inner_opt_builder_train_lr in inner_opt_builder_train_lr_values:
            for inner_opt_builder_test_lr in inner_opt_builder_test_lr_values:
                for outer_lr in outer_lr_values:
                    for num_inner_steps in num_inner_steps_values:
                        for num_epochs in num_epochs_values:
                            args.init_inner_lr = init_inner_lr
                            args.inner_opt_builder_train_lr = inner_opt_builder_train_lr
                            args.inner_opt_builder_test_lr = inner_opt_builder_test_lr
                            args.outer_lr = outer_lr
                            args.num_inner_steps = num_inner_steps
                            args.num_epochs = num_epochs
                            args.output_dir = f'MSR/augment/output/ConvNet_with_Dropout/TESTCUDA_init_inner_lr={init_inner_lr}_train_lr={inner_opt_builder_train_lr}_test_lr={inner_opt_builder_test_lr}_outer_lr={outer_lr}_steps={num_inner_steps}'
                            os.makedirs(args.output_dir, exist_ok=True)
                            main(args)


    main()

