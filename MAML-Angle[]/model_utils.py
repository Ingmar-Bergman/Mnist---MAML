import argparse
import time

import pandas as pd
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('bmh')
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import StratifiedKFold

import torch
import torch.nn.functional as F

import os

import higher



class model_utils:
    def train(db, net, device, meta_opt, scheduler, epoch, log, inner_opt_train_lr, n_inner_iter):

        #x_spt a la forme (task_num, n_way*k_spt, 1, h, w). En effet, pour chaque batch (task_num en tout), on a n_way*k_spt images de taille (1, h, w)
        net.train()
        n_train_iter = db.x_train.shape[0] // db.batchsz
        print(f"n_train_iter: {n_train_iter}")

        for batch_idx in range(n_train_iter):
            start_time = time.time()
            # Sample a batch of support and query images and labels.
            x_spt, y_spt, x_qry, y_qry, _ = db.next()
            x_spt, y_spt = x_spt.to(device), y_spt.to(device)
            x_qry, y_qry = x_qry.to(device), y_qry.to(device)


            task_num, setsz, c_, h, w = x_spt.size()
            querysz = x_qry.size(1)

            # TODO: Maybe pull this out into a separate module so it
            # doesn't have to be duplicated between `train` and `test`?

            # Initialize the inner optimizer to adapt the parameters to
            # the support set.
            inner_opt = torch.optim.SGD(net.parameters(), lr=inner_opt_train_lr)

            qry_losses = []
            qry_accs = []
            meta_opt.zero_grad()

            
            for i in range(task_num): #Number of tasks in the batch
                with higher.innerloop_ctx(
                    net, inner_opt, copy_initial_weights=False
                ) as (fnet, diffopt):
                    # Optimize the likelihood of the support set by taking
                    # gradient steps w.r.t. the model's parameters.
                    # This adapts the model's meta-parameters to the task.
                    # higher is able to automatically keep copies of
                    # your network's parameters as they are being updated.
                    for _ in range(n_inner_iter):

                        spt_logits = fnet(x_spt[i]) 
                        spt_loss = F.cross_entropy(spt_logits, y_spt[i].long()) 
                        diffopt.step(spt_loss)  #gradient descent on the support set

                    # The final set of adapted parameters will induce some
                    # final loss and accuracy on the query dataset.
                    # These will be used to update the model's meta-parameters.
                    qry_logits = fnet(x_qry[i])
                    qry_loss = F.cross_entropy(qry_logits, y_qry[i].long())
                    qry_losses.append(qry_loss.detach())
                    qry_acc = (qry_logits.argmax(
                        dim=1) == y_qry[i]).sum().item() / querysz
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

            scheduler.step(epoch + batch_idx / n_train_iter)


    def test(db, net, device, epoch, log, inner_opt_test_lr, n_inner_iter):
        # Crucially in our testing procedure here, we do *not* fine-tune
        # the model during testing for simplicity.
        # Most research papers using MAML for this task do an extra
        # stage of fine-tuning here that should be added if you are
        # adapting this code for research.
        net.train()
        n_test_iter = db.x_test.shape[0] // db.batchsz



        qry_losses = []
        qry_accs = []

        # all_preds = []
        # all_labels = []


        if db.x_test.shape[0] < db.batchsz:
            print("Not enough data to test")
            raise ValueError("Not enough data to test")  # Raise exception if not enough data


        for batch_idx in range(n_test_iter):
            x_spt, y_spt, x_qry, y_qry, original_class_mapping  = db.next('test') #confusion suppr dernier lement
            x_spt, y_spt = x_spt.to(device), y_spt.to(device)
            x_qry, y_qry = x_qry.to(device), y_qry.to(device)

            # #print original_class_mapping
            # print(original_class_mapping)




            task_num, setsz, c_, h, w = x_spt.size()
            querysz = x_qry.size(1)

            # TODO: Maybe pull this out into a separate module so it
            # doesn't have to be duplicated between `train` and `test`?
            inner_opt = torch.optim.SGD(net.parameters(), lr=inner_opt_test_lr)

            for i in range(task_num):
                with higher.innerloop_ctx(net, inner_opt, track_higher_grads=False) as (fnet, diffopt):
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
                    
                    # #matrice de confusion
                    # all_preds.append(qry_logits.argmax(dim=1).cpu().numpy())
                    # all_labels.append([original_class_mapping[label] for label in y_qry[i].cpu().numpy()])
                    

        qry_losses = torch.cat(qry_losses).mean().item()
        qry_accs = 100. * torch.cat(qry_accs).float().mean().item()
        print(
            f'[Epoch {epoch+1:.2f}] Test Loss: {qry_losses:.2f} | Acc: {qry_accs:.2f}'
        )

        # #matrice de confusion
        # all_preds = np.concatenate(all_preds)
        # all_labels = np.concatenate(all_labels)

        log.append({
            'epoch': epoch + 1,
            'loss': qry_losses,
            'acc': qry_accs,
            'mode': 'test',
            'time': time.time(),
        })

        # return all_preds, all_labels


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


    # def plot_confusion_matrix(true_labels, predictions, class_names, output_dir):
    #     cm = confusion_matrix(true_labels, predictions)
    #     print("Confusion Matrix:")
    #     print(cm)
    #     print("Class Names:")
    #     print(class_names)
    #     cmd = ConfusionMatrixDisplay(cm, display_labels=class_names)
    #     cmd.plot(cmap=plt.cm.Blues)
    #     plt.xticks(rotation=45)
    #     plt.title('Confusion Matrix')
    #     plt.tight_layout()
    #     plt.savefig(f'{output_dir}/confusion_matrix.png')
    #     plt.close()