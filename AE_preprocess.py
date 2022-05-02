import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import _pickle as cPickle
import os
import argparse
import json

import random
from PIL import Image
from torch.autograd import Variable
import pandas as pd


class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        # Input size: [batch, 3, 32, 32]
        # Output size: [batch, 3, 32, 32]
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 12, 4, stride=2, padding=1),            # [batch, 12, 16, 16]
            nn.ReLU(),
            nn.Conv2d(12, 24, 4, stride=2, padding=1),           # [batch, 24, 8, 8]
            nn.ReLU(),
			nn.Conv2d(24, 48, 4, stride=2, padding=1),           # [batch, 48, 4, 4]
            nn.ReLU(),
# 			nn.Conv2d(48, 96, 4, stride=2, padding=1),           # [batch, 96, 2, 2]
#             nn.ReLU(),
        )
        self.decoder = nn.Sequential(
#             nn.ConvTranspose2d(96, 48, 4, stride=2, padding=1),  # [batch, 48, 4, 4]
#             nn.ReLU(),
			nn.ConvTranspose2d(48, 24, 4, stride=2, padding=1),  # [batch, 24, 8, 8]
            nn.ReLU(),
			nn.ConvTranspose2d(24, 12, 4, stride=2, padding=1),  # [batch, 12, 16, 16]
            nn.ReLU(),
            nn.ConvTranspose2d(12, 3, 4, stride=2, padding=1),   # [batch, 3, 32, 32]
            nn.Tanh(),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded


# functions to show an image
def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))

# Return only images of certain class (eg. plane = class 0)
def get_same_index(target, label):
    label_indices = []
    for i in range(len(target)):
        if target[i] == label:
            label_indices.append(i)
    return label_indices

def unpickle(file):
    
    with open(file, 'rb') as fo:
        dict = cPickle.load(fo, encoding='latin1')
    return dict


class cifar_dataset(Dataset):
    def __init__(self, dataset, root_dir, transform, mode, noise_file=''):

        self.transform = transform
        self.mode = mode
        self.transition = {0: 0, 2: 0, 4: 7, 7: 7, 1: 1, 9: 1, 3: 5, 5: 3, 6: 6,
                           8: 8}  # class transition for asymmetric noise for cifar10
        # generate asymmetric noise for cifar100
        self.transition_cifar100 = {}
        nb_superclasses = 20
        nb_subclasses = 5
        base = [1, 2, 3, 4, 0]
        for i in range(nb_superclasses * nb_subclasses):
            self.transition_cifar100[i] = int(base[i % 5] + 5 * int(i / 5))

        if self.mode == 'test':
            if dataset == 'cifar10':
                test_dic = unpickle('%s/test_batch' % root_dir)
                self.test_data = test_dic['data']
                self.test_data = self.test_data.reshape((10000, 3, 32, 32))
                self.test_data = self.test_data.transpose((0, 2, 3, 1))
                self.test_label = test_dic['labels']
            elif dataset == 'cifar100':
                test_dic = unpickle('%s/test' % root_dir)
                self.test_data = test_dic['data']
                self.test_data = self.test_data.reshape((10000, 3, 32, 32))
                self.test_data = self.test_data.transpose((0, 2, 3, 1))
                self.test_label = test_dic['fine_labels']
        else:
            train_data = []
            train_label = []
            if dataset == 'cifar10':
                for n in range(1, 6):
                    dpath = '%s/data_batch_%d' % (root_dir, n)
                    data_dic = unpickle(dpath)
                    train_data.append(data_dic['data'])
                    train_label = train_label + data_dic['labels']
                train_data = np.concatenate(train_data)
            elif dataset == 'cifar100':
                train_dic = unpickle('%s/train' % root_dir)
                train_data = train_dic['data']
                train_label = train_dic['fine_labels']
                # print(train_label)
                # print(len(train_label))
            train_data = train_data.reshape((50000, 3, 32, 32))
            train_data = train_data.transpose((0, 2, 3, 1))

            noise_label = json.load(open(noise_file, "r"))

            if self.mode == 'train':
                self.train_data = train_data
                self.noise_label = noise_label
                self.clean_label = train_label

    def __getitem__(self, index):
        if self.mode == 'train':
            img, target = self.train_data[index], self.noise_label[index]
            img = Image.fromarray(img)
            img = self.transform(img)
            return img, target, index
        elif self.mode == 'test':
            img, target = self.test_data[index], self.test_label[index]
            img = Image.fromarray(img)
            img = self.transform(img)
            return img, target

    def __len__(self):
        if self.mode != 'test':
            return len(self.train_data)
        else:
            return len(self.test_data)


class cifar_dataloader():
    def __init__(self, dataset, batch_size, num_workers, root_dir, noise_file=''):
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.root_dir = root_dir
        self.noise_file = noise_file
        if self.dataset == 'cifar10':
            self.transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
            self.transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
        elif self.dataset == 'cifar100':
            self.transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.507, 0.487, 0.441), (0.267, 0.256, 0.276)),
            ])
            self.transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.507, 0.487, 0.441), (0.267, 0.256, 0.276)),
            ])

    def run(self, mode):
        if mode == 'train':
            train_dataset = cifar_dataset(dataset=self.dataset,
                                          root_dir=self.root_dir, transform=self.transform_train, mode="train",
                                          noise_file=self.noise_file)
            trainloader = DataLoader(
                dataset=train_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers)
            return trainloader, np.asarray(train_dataset.noise_label), np.asarray(train_dataset.clean_label)

        elif mode == 'test':
            test_dataset = cifar_dataset(dataset=self.dataset,
                                         root_dir=self.root_dir, transform=self.transform_test, mode='test')
            test_loader = DataLoader(
                dataset=test_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers)
            return test_loader



def main():

    parser = argparse.ArgumentParser(description="Train Autoencoder")
    parser.add_argument("--valid", action="store_true", default=False,
                        help="Perform validation only.")
    args = parser.parse_args()

    # todo: prepare the training data and test data
    gpuid = '1'

    os.environ['CUDA_VISIBLE_DEVICES'] = gpuid
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('We are using', device)

    # test the custom loaders for CIFAR
    dataset = 'cifar10'  # either cifar10 or cifar100
    data_path = 'data/cifar-10-batches-py'  # path to the data file (don't forget to download the feature data and also put the noisy label file under this folder)

    batch_size = 20
    loader = cifar_dataloader(dataset, batch_size=batch_size,
                            num_workers=1,
                            root_dir=data_path,
                            noise_file='%s/cifar10_noisy_labels_task1.json' % (data_path))

    trainloader, noisy_labels, clean_labels = loader.run('train')
    testloader = loader.run('test')

    classes = ('plane', 'car', 'bird', 'cat',
            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    clean_train_indices = []  # Clean train sample indices
    for label_class in range(10):
        trainloader, noisy_labels, clean_labels = loader.run('train')
        # get the images of a certain class
        print("AE for class ", classes[label_class])
        #label_class = 2  # car
        # Get indices of label_class
        train_indices = get_same_index(noisy_labels, label_class)
        print("The number of %s " % classes[label_class], len(train_indices))

        label_class_set = torch.utils.data.Subset(trainloader.dataset, train_indices)

        trainloader = torch.utils.data.DataLoader(dataset=label_class_set, shuffle=True,
                                            batch_size=batch_size, drop_last=True)

        # get some random training images
        dataiter = iter(trainloader)
        images, labels, c_labels = dataiter.next()

        # dataiter = iter(all_trainloader)
        # images, n_labels, c_labels = dataiter.next()

        print(torch.min(images), torch.max(images))

        # show images
        fig = plt.figure(figsize=(25, 4))
        for idx in np.arange(20):
            ax = fig.add_subplot(2, int(20/2), idx+1, xticks=[], yticks=[])
            imshow(images[idx])
            ax.set_title(classes[labels[idx]])
        plt.show()
        # print labels
        print(' '.join(f'{classes[labels[j]]:5s}' for j in range(batch_size)))
        
        for j in range(batch_size):
            print(classes[clean_labels[c_labels[j]]])

        exit(0)
        # Initializing the Auto-Encoder
        autoencoder = Autoencoder()

        print("============== Encoder ==============")
        print(autoencoder.encoder)
        print("============== Decoder ==============")
        print(autoencoder.decoder)
        print("")

        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(autoencoder.parameters(),
                                lr=1e-3, 
                                weight_decay=1e-5)
        
        # Training the Auto-Encoder
        num_epochs = 30
        outputs = []
        last_loss = 0
        for epoch in range(num_epochs):
            for (img, _, _) in trainloader:
                # img = img.reshape(-1, 28*28) # -> use for Autoencoder_Linear
                encoded, decoded = autoencoder(img)
                loss = criterion(decoded, img)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            print(f'Epoch:{epoch+1}, Loss:{loss.item():.4f}')
            last_loss = loss.item()
            outputs.append((epoch, img, decoded))


        # Plotting the images and their corresponding reconstructed images for some of epochs
        for k in range(0, num_epochs, 4):
            fig = plt.figure(figsize=(7,4.5))
            imgs = outputs[k][1]
            recon = outputs[k][2]

            imgs = torch.stack([imgs, recon], dim=1).flatten(0,1)
            grid = torchvision.utils.make_grid(imgs, nrow=8, normalize=True, value_range=(-1,1))
            grid = grid.permute(1, 2, 0)
            plt.title(f"Results from epoch {k+1}")
            plt.imshow(grid)
            plt.axis('off')
            #plt.show()
            plt.savefig("images/%s-%s.png" % (classes[label_class], k))

        # Computing the reconstruction loss of single images and record their dataset indices for later filtering!
        print("Checking single image's reconstruction costs")
        print("The last loss ", last_loss)
        outputs_high_loss = []
        for (img, _, idx) in trainloader:
            # img = img.reshape(-1, 28*28) # -> use for Autoencoder_Linear
            #print(type(img), len(img), type(img[10]), len(img[10]), idx)
            for sample_i in range(len(img)):
                x = img[sample_i]
                dataset_idx = idx[sample_i]
                encoded, x_hat = autoencoder(x[None, ...])
                loss = F.mse_loss(x[None, ...], x_hat, reduction="none")
                loss = loss.mean(dim=[1,2,3])
                #print(loss, dataset_idx, noisy_labels[dataset_idx], clean_labels[dataset_idx])
                if loss.item() > 1.0*last_loss:
                    outputs_high_loss.append((x[None, ...], x_hat, noisy_labels[dataset_idx], clean_labels[dataset_idx], dataset_idx.item(), loss.item()))

        total_to_be_removed = len(outputs_high_loss)
        total_to_be_removed_identified_correctly = len([x[3] for x in outputs_high_loss if x[3]!=label_class])
        correct_percentage = total_to_be_removed_identified_correctly / total_to_be_removed
        print("Stats of removal ", total_to_be_removed, total_to_be_removed_identified_correctly, correct_percentage)

        noisy_ds_indices = [x[4] for x in outputs_high_loss]
        train_indices = get_same_index(noisy_labels, label_class)
        clean_class_train_indices = list(set(train_indices).difference(set(noisy_ds_indices)))

        # store clean train indices
        print(len(clean_class_train_indices))
        print(len(clean_train_indices))
        clean_train_indices.extend(clean_class_train_indices)
        print(len(clean_train_indices))

        clean_set = torch.utils.data.Subset(trainloader.dataset, clean_class_train_indices)
        print("Size after removal ", len(clean_set))

        trainloader = torch.utils.data.DataLoader(dataset=clean_set, shuffle=True,
                                            batch_size=batch_size, drop_last=True)
        
        fig = plt.figure(figsize=(7,4.5))
        highest_losses = sorted(outputs_high_loss, key=lambda x: x[5], reverse=True)
        imgs = [x[0] for x in highest_losses[:20]]

        recon = [x[1] for x in highest_losses[:20]]
        
        #[val for pair in zip(imgs, recon) for val in pair]

        print([(classes[x[2]], classes[x[3]], x[5]) for x in highest_losses[:20]])

        imgs = torch.stack([val for pair in zip(imgs, recon) for val in pair], dim=1).flatten(0,1)
        grid = torchvision.utils.make_grid(imgs, nrow=8, normalize=True, value_range=(-1,1))
        grid = grid.permute(1, 2, 0)
        #plt.title(f"Results from epoch {k+1}")
        plt.imshow(grid)
        plt.axis('off')
        #plt.show()
        plt.savefig("images/top20-recon-loss-%s.png" % classes[label_class])

    #clean_train_indices
    df = pd.DataFrame(clean_train_indices, columns =['indices'])
    #print(df.head(5))
    df.to_csv("images/clean_indices.csv", index=False)
    

if __name__ == '__main__':
    main()
