import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import numpy as np
import os
import random

device = "cuda:0" if torch.cuda.is_available() else "cpu"

# transform = transforms.Compose([
#     lambda x: Image.open(x).convert('L'),  # Open image and convert to grayscale
#     lambda x: x.resize((28, 28)),  # Resize image to 28x28
#     lambda x: np.reshape(x, (28, 28, 1)),  # Reshape image by adding channel dimension
#     lambda x: np.transpose(x, [2, 0, 1]),  # Transpose dimensions to (channel, height, width)
#     lambda x: x / 255.0  # Normalize pixel values to [0, 1]
# ])

# figure = plt.figure(figsize=(8, 8))
# cols, rows = 3, 3
# for i in range(1, cols * rows + 1):
#     sample_idx = torch.randint(len(training_data), size=(1,)).item()
#     img, label = training_data[sample_idx]
#     figure.add_subplot(rows, cols, i)
#     plt.title(labels_map[label])
#     plt.axis("off")
#     plt.imshow(img.squeeze(), cmap="gray")
# plt.show()


import  torchvision.transforms as transforms
from    PIL import Image
import  numpy as np

import torch
import  torch.utils.data as data
import  os
import  os.path
import  errno

#download=True ou false ?



#Faudrait shuffle et mélanger au début le training set et le data set, car des classes sur lesquelles on s'est déjà entraînés seront sur le test set

#SHUFFLE SHUFFLE
class RotatedMNIST(Dataset):
    def __init__(self, root, train=True, transform=None, download=False, num_classes= 20, num_elements_per_class= 20):
        self.train_data = datasets.MNIST(root=root, train=True, download=download)
        self.test_data = datasets.MNIST(root=root, train=False, download=download)

        # #shuffle self.train_data
        # indices = list(range(len(self.train_data)))
        # random.shuffle(indices)
        # self.train_data = torch.utils.data.Subset(self.train_data, indices)

        # #shuffle self.train_data
        # indices = list(range(len(self.test_data)))
        # random.shuffle(indices)
        # self.train_data = torch.utils.data.Subset(self.test_data, indices)
        
        # Concatenate train and test datasets
        self.transform = transform

        # Paramètres par défaut
        self.num_elements_per_class = num_elements_per_class
        self.nombre_rotations = num_classes

        # Générer les données transformées
        self.transformed_data = self.generate_transformed_data(self.num_elements_per_class, self.nombre_rotations)

        print("Number of classes:", len(set([label for _, label in self.transformed_data])))
        print("Length of self.transformed_data:", len(self.transformed_data))

        new_test_data= self.select_test_data(self.num_elements_per_class)

        #print informatioons about self.transformed_data
        print("Length of self.transformed_data:", len(new_test_data))
        #number of classes 
        print("Number of classes_new_test_data:", len(set([label for _, label in new_test_data])))



        # Concatenate transformed_data and  self.test
        self.transformed_data = self.transformed_data + new_test_data



        # print some images with their labels with mathplotlib
        # figure = plt.figure(figsize=(8, 8))
        # cols, rows = 3, 3
        # for i in range(1, cols * rows + 1):
        #     sample_idx = torch.randint(len(self.transformed_data), size=(1,)).item()
        #     img, label = self.transformed_data[sample_idx]
        #     figure.add_subplot(rows, cols, i)
        #     plt.title(label)
        #     plt.axis("off")
        #     plt.imshow(img, cmap="gray")
        # plt.show()
        


    #Transforme le self.train_data
    def generate_transformed_data(self, num_elements_per_class, nombre_rotations):
        # Liste pour stocker les datasets transformés
        transformed_datasets = []
        self.rotations = np.linspace(360//nombre_rotations, 360, nombre_rotations, endpoint=False).astype(int)
        print(self.rotations)



        count = 0
        temp_images = []
        rotation_index = 0  # Pour suivre l'index des rotations

        for i in range(len(self.train_data)):
            if rotation_index >= len(self.rotations):
                break  # Arrêter l'itération si toutes les rotations ont été utilisées

            image, _ = self.train_data[i]  # Ignorer le label original
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
    
    def select_test_data(self, num_elements_per_class):
        class_dict = {}
        for image, label in self.test_data:
            if label in class_dict:
                if len(class_dict[label]) < num_elements_per_class:
                    class_dict[label].append(image)
            else:
                class_dict[label] = [image]

        selected_data = []
        for label, images in class_dict.items():
            images = images[:num_elements_per_class]
            images = [np.expand_dims(img, axis=0) for img in images]
            selected_data.extend([(img, label) for img in images])

        return selected_data


    def __len__(self):
        return len(self.transformed_data)

    def __getitem__(self, idx):
        image, label = self.transformed_data[idx]
        if self.transform:
            image = self.transform(image)
        return image, label
    

class MNISTNShot:

    #Dans les arguments, enlever training et test et mettre que data, ensuite décommenter les commentaires

    def __init__(self, dataset, batchsz, n_way, k_shot, k_query, imgsz, num_elements_per_class, perc_training_set, device=None):
        """
        Different from mnistNShot, the
        :param root:
        :param batchsz: task num
        :param n_way:
        :param k_shot:
        :param k_qry:
        :param imgsz:
        """

        self.resize = imgsz
        self.device = device
        self.num_elements_per_class = num_elements_per_class

        assert (k_shot + k_query) <=num_elements_per_class

        # Create a dictionary under the form {label:img1, img2..., num_elements_per_class imgs, label2: img1, img2,...}
        temp = dict()
        for i in range(len(dataset.transformed_data)):
            label = dataset.transformed_data[i][1]  # Convert numpy array to tuple
            image = np.array(dataset.transformed_data[i][0])  # Assuming dataset.data[i] is already numpy array
            if label in temp:
                temp[label].append(image)
            else:
                temp[label] = [image]
            
        self.x=[]

        # print first element of temp.items()
        # print("First element of temp.items():", list(temp.items())[0])


        #iterate on temp.items()
        for label,imgs in temp.items():
            self.x.append(np.array(imgs)) #voir si np.array est utile ici
        
        # as different class may have different number of imgs
        
        self.x = np.stack(self.x).astype(float)  # [[n imgs],..., xx classes in total]
        # each character contains num_elements_per_class imgs
        
        print('data shape:', self.x.shape)  #{10, num_elements_per_class, 1, imgsz, imgsz]
        temp = []  # Free memory

        
        self.x_train, self.x_test = self.x[:-10], self.x[-10:]

        #get the labels of self.x_test



        # self.normalization() #Regarder normalisation sur le training set ou le batch !

        self.batchsz = batchsz   #Nombre de tâches traitées en même temps dans un batch
        
        self.n_cls = self.x.shape[0]  # Number of unique classes

        print("Number of unique classes:", self.n_cls)
        
        print(self.n_cls)
        self.n_way = n_way  # n way
        self.k_shot = k_shot  # k shot
        self.k_query = k_query  # k query


        # save pointer of current read batch in total cache
        self.indexes = {"train": 0, "test": 0}
        self.datasets = {"train": self.x_train, "test": self.x_test}  # original data cached
        print("DB: train", self.x_train.shape, "test", self.x_test.shape)

        self.datasets_cache = {"train": self.load_data_cache(self.datasets["train"]),  # current epoch data cached
                               "test": self.load_data_cache(self.datasets["test"])}
        
        # #print 10 labels of self.datasets_cache["train"]
        # print("10 labels of self.datasets_cache[\"train\"]:", self.datasets_cache["train"][0][1][0:10])
        


    # def normalization(self):
    #     """
    #     Normalizes our data, to have a mean of 0 and sdt of 1
    #     """
    #     self.mean = np.mean(self.x_train)
    #     self.std = np.std(self.x_train)
    #     self.max = np.max(self.x_train)
    #     self.min = np.min(self.x_train)
    #     # print("before norm:", "mean", self.mean, "max", self.max, "min", self.min, "std", self.std)
    #     self.x_train = (self.x_train - self.mean) / self.std
    #     self.x_test = (self.x_test - self.mean) / self.std

    #     self.mean = np.mean(self.x_train)
    #     self.std = np.std(self.x_train)
    #     self.max = np.max(self.x_train)
    #     self.min = np.min(self.x_train)

    #     print("after norm:", "mean", self.mean, "max", self.max, "min", self.min, "std", self.std)
    

    # def create_data_cache(self, dataset):
    #     data_cache = []
    #     labels_cache = []
    #     for img, (label, angle) in dataset:
    #         img = np.array(img)
    #         data_cache.append(img)
    #         labels_cache.append(label)
    #     data_cache = np.array(data_cache).astype(np.float32)
    #     labels_cache = np.array(labels_cache).astype(np.int)
    #     return data_cache, labels_cache
    
    
    
    
    def load_data_cache(self, data_pack):
        """
        Collects several batches data for N-shot learning
        :param data_pack: [cls_num, 20, 84, 84, 1]
        :return: A list with [support_set_x, support_set_y, target_x, target_y] ready to be fed to our networks
        """
        #  take 5 way 1 shot as example: 5 * 1
        setsz = self.k_shot * self.n_way #Nombre d'images en tout
        querysz = self.k_query * self.n_way # Nombre total d'images dans le query set
        data_cache = []

        # print('preload next 50 caches of batchsz of batch.')
        for sample in range(10):  # num of episodes

            x_spts, y_spts, x_qrys, y_qrys = [], [], [], []
            original_class_mapping = {} #confusion


            for i in range(self.batchsz):  # one batch means one set #numero task

                x_spt, y_spt, x_qry, y_qry = [], [], [], []

                #shape[0] correspond aux nombres de classes
                selected_cls = np.random.choice(data_pack.shape[0], self.n_way, False) # Sélectionne de manière aléatoire "n-way" classes parmi les data_pack.shape[0]
                
                for j, cur_class in enumerate(selected_cls):
                    original_class_mapping[j] = cur_class #confusion
                    selected_img = np.random.choice(self.num_elements_per_class, self.k_shot + self.k_query, False) # Sélectionne de manière aléatoire "self.k_shot + self.k_query" images parmi les num_elements_per_class de la classe courante
                    
                    # meta-training and meta-test / 
                    x_spt.append(data_pack[cur_class][selected_img[:self.k_shot]]) #x_spt est converti en un tableau numpy et remodelé pour avoir la forme [self.n_way * self.k_shot, 1, self.resize, self.resize]
                    x_qry.append(data_pack[cur_class][selected_img[self.k_shot:]])
                    y_spt.append([j for _ in range(self.k_shot)]) # Associe les labels appropriés (index de la classe) 0,..,n_way-1
                    y_qry.append([j for _ in range(self.k_query)])

                # shuffle inside a batch
                perm = np.random.permutation(self.n_way * self.k_shot)
                x_spt = np.array(x_spt).reshape(self.n_way * self.k_shot, 1, self.resize, self.resize)[perm]
                y_spt = np.array(y_spt).reshape(self.n_way * self.k_shot)[perm]
                perm = np.random.permutation(self.n_way * self.k_query)
                x_qry = np.array(x_qry).reshape(self.n_way * self.k_query, 1, self.resize, self.resize)[perm]
                y_qry = np.array(y_qry).reshape(self.n_way * self.k_query)[perm]

                # append [sptsz, 1, 84, 84] => [b, setsz, 1, 84, 84]
                x_spts.append(x_spt)
                y_spts.append(y_spt)
                x_qrys.append(x_qry)
                y_qrys.append(y_qry)

            # Convertit les listes en tableaux numpy et les reshape aux dimensions nécessaires
            # [b, setsz, 1, 84, 84]
            x_spts = np.array(x_spts).astype(np.float32).reshape(self.batchsz, setsz, 1, self.resize, self.resize)
            y_spts = np.array(y_spts).astype(np.int).reshape(self.batchsz, setsz)
            # [b, qrysz, 1, 84, 84]
            x_qrys = np.array(x_qrys).astype(np.float32).reshape(self.batchsz, querysz, 1, self.resize, self.resize)
            y_qrys = np.array(y_qrys).astype(np.int).reshape(self.batchsz, querysz)

            x_spts, y_spts, x_qrys, y_qrys = [
                torch.from_numpy(z).to(self.device) for z in
                [x_spts, y_spts, x_qrys, y_qrys]
            ]

            data_cache.append([x_spts, y_spts, x_qrys, y_qrys, original_class_mapping]) #confusion supprimer le dernier élément

        return data_cache

    def next(self, mode='train'):
        """
        Gets next batch from the dataset with name.
        :param mode: The name of the splitting (one of "train", "val", "test")
        :return:
        """
        # update cache if indexes is larger cached num
        if self.indexes[mode] >= len(self.datasets_cache[mode]):
            self.indexes[mode] = 0
            self.datasets_cache[mode] = self.load_data_cache(self.datasets[mode])

        next_batch = self.datasets_cache[mode][self.indexes[mode]]
        self.indexes[mode] += 1

        return next_batch
    
# transform = transforms.Compose([
# lambda x: Image.open(x).convert('L'),  # Open image and convert to grayscale
# lambda x: x.resize((28, 28)),  # Resize image to 28x28
# lambda x: np.reshape(x, (28, 28, 1)),  # Reshape image by adding channel dimension
# lambda x: np.transpose(x, [2, 0, 1]),  # Transpose dimensions to (channel, height, width)
# lambda x: x / 255.0  # Normalize pixel values to [0, 1]
# ])


# # Vérification
# print(f"Nombre total d'images: {len(dataset)}")
# for i in range(0, len(dataset), 40):
#     print(f"Label de la tâche {i//40 + 1}: {dataset[i][1]}")
# #print the image dataset[0] with label with matplolib
# plt.imshow(dataset[0][0].squeeze(), cmap="gray")
# plt.title(dataset[0][1])
# plt.show()


