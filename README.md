# Mnist - MAML

This project implements a N-way k-shot on MNIST dataset.
There are num_classes classes in the training set which consists of MNIST samples rotated by an angle.
The test set consists of the real dataset MNIST with the true labels.



### MSR (ConvNet sans dropout)

Avec les paramètres par défaut mais avec un num_inner_steps=7 :  

![demonstrative figure](MSR/Augment/output/ConvNet_Sans_Dropout/TEST_init_inner_lr=0.1_train_lr=0.1_test_lr=0.1_outer_lr=0.001_steps=7/ConvNet.png)
![demonstrative figure](MSR/Augment/output/ConvNet_Sans_Dropout/TEST_init_inner_lr=0.1_train_lr=0.1_test_lr=0.1_outer_lr=0.001_steps=7/confusion_matrix.png)



### MAML (ConvNet (avec dropout))

Avec les paramètres par défaut mais avec un num_inner_steps=7 : 

![demonstrative figure](MSR/Augment/output/ConvNet_with_Dropout/init_inner_lr=0.1_train_lr=0.1_test_lr=0.1_outer_lr=0.001_steps=7)
![demonstrative figure](MSR/Augment/output/ConvNet_with_Dropout/init_inner_lr=0.1_train_lr=0.1_test_lr=0.1_outer_lr=0.001_steps=7/confusion_matrix.png)


### MAML (ConvNet sans dropout)

Avec les paramètres par défaut : 

![demonstrative figure](MAML/output/ConvNet_Sans_Dropout/par_def)

### MAML (SimpleNN)

Avec les paramètres par défaut : 

![demonstrative figure](MAML/output/SimpleNN/ConvNet_Sans_Dropout-1.png)

