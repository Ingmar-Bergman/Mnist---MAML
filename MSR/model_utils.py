
import argparse
import time

import pandas as pd
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('bmh')

import torchvision.transforms as transforms


import torch
import torch.nn.functional as F

import os
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score
import higher





class model_utils:
    def train(db, net, device, inner_opt_builder, meta_opt, scheduler, epoch, log, args):
        net.train()
        n_train_iter = db.x_train.shape[0] // db.batchsz

        augmentation = transforms.Compose([
            transforms.RandomResizedCrop(28, scale=(0.8, 1.0)),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(30),
        ])


        for batch_idx in range(n_train_iter):
            start_time = time.time()
            # Sample a batch of support and query images and labels.
            x_spt, y_spt, x_qry, y_qry,_ = db.next('train')
            x_spt, y_spt, x_qry, y_qry = x_spt.to(device), y_spt.to(device), x_qry.to(device), y_qry.to(device)
            print(f"x_spt: {x_spt.shape}, y_spt: {y_spt.shape}")
            print(f"x_qry: {x_qry.shape}, y_qry: {y_qry.shape}")

            task_num, setsz, c_, h, w = x_spt.size()
            querysz = x_qry.size(1)

            # TODO: Maybe pull this out into a separate module so it
            # doesn't have to be duplicated between `train` and `test`?

            # Initialize the inner optimizer to adapt the parameters to
            # the support set.
            n_inner_iter = args.num_inner_steps #
            # inner_opt = torch.optim.SGD(net.parameters(), lr=1e-1) #remplacé par ADAM ?

            qry_losses = []
            qry_accs = []

            x_qry_combined = []
            y_qry_combined = []

            meta_opt.zero_grad()

            for i in range(task_num):  # Number of tasks in the batch
                # Apply data augmentation to the query set
                x_qry_aug = torch.cat([augmentation(x.unsqueeze(0)) for x in x_qry[i]], dim=0)
                # Concatenate original and augmented query sets for each task
                x_qry_combined.append(torch.cat((x_qry[i], x_qry_aug), dim=0))
                y_qry_combined.append(torch.cat((y_qry[i], y_qry[i]), dim=0))


            # Stack the combined query sets to match the desired shape
            x_qry_combined = torch.stack(x_qry_combined)
            y_qry_combined = torch.stack(y_qry_combined)

            x_qry_combined, y_qry_combined = x_qry_combined.to(device), y_qry_combined.to(device)
            
            for i in range(task_num): #Number of tasks in the batch


                with higher.innerloop_ctx(
                    net, inner_opt_builder.inner_opt, copy_initial_weights=False, override=inner_opt_builder.overrides
                ) as (fnet, diffopt):

                    # Optimize the likelihood of the support set by taking
                    # gradient steps w.r.t. the model's parameters.
                    # This adapts the model's meta-parameters to the task.
                    # higher is able to automatically keep copies of
                    # your network's parameters as they are being updated.
                    for _ in range(n_inner_iter):
                        #print the label of x_spt[i]
                        # print(f"y_spt[{i}]:", y_spt[i])
                        
                        #VERIFIER DIMENSION DE X_SPT
                        
                        spt_logits = fnet(x_spt[i])
                        # print(f"spt_logits: {spt_logits}")
                        spt_loss = F.cross_entropy(spt_logits, y_spt[i].long())
                        # print(f"spt_loss: {spt_loss}")
                        diffopt.step(spt_loss)
                        # y= x_spt[i].view(args.n_way * args.k_spt, 1, 28, 28)
                        # print(f"y_shape{y.shape}")  

                        # spt_logits = fnet(x_spt[i].view(args.n_way * args.k_spt, 1, 28, 28))
                        # spt_loss = F.cross_entropy(spt_logits, y_spt[i].view(args.n_way * args.k_spt).long())
                        # diffopt.step(spt_loss)
                        

                    # The final set of adapted parameters will induce some
                    # final loss and accuracy on the query dataset.
                    # These will be used to update the model's meta-parameters.
                    # qry_logits = fnet(x_qry[i])
                    # qry_loss = F.cross_entropy(qry_logits, y_qry[i].long())
                    # qry_losses.append(qry_loss.detach())
                    # qry_acc = (qry_logits.argmax(
                    #     dim=1) == y_qry[i]).sum().item() / querysz
                    # qry_accs.append(qry_acc)


                    # REVENIR A AVANT, JUSTE REMETTRE X_QRY, Y_QRY
                    #print x_qry_combined[i] shape
                    # print(f"x_qry_combined[{i}]: {x_qry_combined[i].shape}") #torch.Size([1, 28, 28]

                    #  print x_qry[i] shape
                    # print(f"x_qry[{i}]: {x_qry[i].shape}") #torch.Size([75, 1, 28, 28])
 

                    qry_logits = fnet(x_qry_combined[i].view(args.n_way * args.k_qry*2, 1, 28, 28))
                    qry_loss = F.cross_entropy(qry_logits, y_qry_combined[i].view(args.n_way * args.k_qry*2).long())
                    qry_losses.append(qry_loss.detach())
                    qry_acc = (qry_logits.argmax(dim=1) == y_qry_combined[i].view(args.n_way * args.k_qry*2)).sum().item() / (2*args.n_way * args.k_qry)
                    qry_accs.append(qry_acc)

                    # Update the model's meta-parameters to optimize the query
                    # losses across all of the tasks sampled in this batch.
                    # This unrolls through the gradient steps.
                    qry_loss.backward()

            meta_opt.step()
            qry_losses = sum(qry_losses) / task_num
            qry_accs = 100. * sum(qry_accs) / task_num
            i = epoch + float(batch_idx) / n_train_iter
            iter_time = time.time() - start_time
            if batch_idx % 4 == 0:
                print(
                    f'[Epoch {i:.2f}] Train Loss: {qry_losses:.2f} | Acc: {qry_accs:.2f} | Time: {iter_time:.2f}'
                )

            log.append({
                'epoch': i,
                'loss': qry_losses,
                'acc': qry_accs,
                'mode': 'train',
                'time': time.time(),
            })
            
            scheduler.step()


    def test(db, net, device, inner_opt_builder, epoch, log, args):
        # Crucially in our testing procedure here, we do *not* fine-tune
        # the model during testing for simplicity.
        # Most research papers using MAML for this task do an extra
        # stage of fine-tuning here that should be added if you are
        # adapting this code for research.
        net.train()
        n_test_iter = db.x_test.shape[0] // db.batchsz

        if db.x_test.shape[0] < db.batchsz:
            print("Not enough data to test")
            return

        qry_losses = []
        qry_accs = []

        all_true_labels = [] #confusion 
        all_pred_labels = [] #confusion

        for batch_idx in range(n_test_iter):
            x_spt, y_spt, x_qry, y_qry, batch_label_mappings = db.next('test')


            task_num, setsz, c_, h, w = x_spt.size()
            querysz = x_qry.size(1)

            # TODO: Maybe pull this out into a separate module so it
            # doesn't have to be duplicated between `train` and `test`?
            n_inner_iter = args.num_inner_steps
            # inner_opt = torch.optim.SGD(net.parameters(), lr=1e-1) #SGD remplacé par ADAM ?

            for i in range(task_num):
                with higher.innerloop_ctx(net, inner_opt_builder.inner_opt, track_higher_grads=False, override=inner_opt_builder.overrides) as (fnet, diffopt):
                    # Optimize the likelihood of the support set by taking
                    # gradient steps w.r.t. the model's parameters.
                    # This adapts the model's meta-parameters to the task.
                    for _ in range(n_inner_iter):
                        spt_logits = fnet(x_spt[i])
                        spt_loss = F.cross_entropy(spt_logits, y_spt[i].long())
                        diffopt.step(spt_loss)

                    # The query loss and acc induced by these parameters.
                    qry_logits = fnet(x_qry[i]).detach()
                    qry_loss = F.cross_entropy(
                        qry_logits, y_qry[i].long(), reduction='none')
                    qry_losses.append(qry_loss.detach())
                    qry_accs.append(
                        (qry_logits.argmax(dim=1) == y_qry[i]).detach())
                
                # Gestion des étiquettes pour la confusion
                label_mapping = batch_label_mappings[i]

                #convert y_qry[i] and qry_logits to their original values
                true_labels = [label_mapping[y.item()] for y in y_qry[i]]
                pred_labels = [label_mapping[y.item()] for y in qry_logits.argmax(dim=1)]


                all_true_labels.extend(true_labels)
                all_pred_labels.extend(pred_labels)



        qry_losses = torch.cat(qry_losses).mean().item()
        qry_accs = 100. * torch.cat(qry_accs).float().mean().item()
        print(
            f'[Epoch {epoch+1:.2f}] Test Loss: {qry_losses:.2f} | Acc: {qry_accs:.2f}'
        )
        log.append({
            'epoch': epoch + 1,
            'loss': qry_losses,
            'acc': qry_accs,
            'mode': 'test',
            'time': time.time(),
        })

        return all_true_labels, all_pred_labels

    @staticmethod
    def save_results(args, log):
        os.makedirs(args.output_dir, exist_ok=True)
        file_path = os.path.join(args.output_dir, f'results_{args.model_class}.txt')

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
    def plot(log,args):
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

        # Create the directory if it doesn't exist
        os.makedirs(args.output_dir, exist_ok=True)
        
        # Construct the filename based on args
        fname = f'{args.model_class}.png'
        fname = os.path.join(args.output_dir, fname)
        
        print(f'--- Plotting accuracy to {fname}')
        fig.savefig(fname)
        plt.close(fig)

    def compute_and_save_confusion_matrix(true_labels, pred_labels, output_dir):
        conf_matrix = confusion_matrix(true_labels, pred_labels)
        accuracy = accuracy_score(true_labels, pred_labels)


        # Conversion des comptes en pourcentages
        row_sums = conf_matrix.sum(axis=1, keepdims=True)  # Somme des lignes
        conf_matrix_percentage = conf_matrix.astype('float') / row_sums  # Calcul des pourcentages
        #tronquer à 1 chiffres après la virgule
        conf_matrix_percentage = np.round(conf_matrix_percentage, 1)
        
        #print a message if there is Nan
        if np.isnan(conf_matrix_percentage).any():
            print('Warning: NaN values in the confusion matrix. This is likely due to classes with no true positives.')


        conf_matrix_percentage = np.nan_to_num(conf_matrix_percentage * 100)  # Remplace NaN par 0



        print('Confusion Matrix:')
        print(conf_matrix)
        print(f'Overall Accuracy: {accuracy * 100:.2f}%')

        # Plot and save the confusion matrix
        plt.figure(figsize=(20,14))
        sns.heatmap(conf_matrix_percentage, annot=True, fmt='.1f', cmap='Blues')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
        plt.close()