import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

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

# 3x3 convolution
def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                     stride=stride, padding=1, bias=False)

# Residual block
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out

# ResNet
class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=10):
        super(ResNet, self).__init__()
        self.in_channels = 16
        self.conv = conv3x3(3, 16)
        self.bn = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self.make_layer(block, 16, layers[0])
        self.layer2 = self.make_layer(block, 32, layers[1], 2)
        self.layer3 = self.make_layer(block, 64, layers[2], 2)
        self.avg_pool = nn.AvgPool2d(8)
        self.fc = nn.Linear(64, num_classes)

    def make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if (stride != 1) or (self.in_channels != out_channels):
            downsample = nn.Sequential(
                conv3x3(self.in_channels, out_channels, stride=stride),
                nn.BatchNorm2d(out_channels))
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels
        for i in range(1, blocks):
            layers.append(block(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


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

    batch_size = 16
    loader = cifar_dataloader(dataset, batch_size=batch_size,
                            num_workers=1,
                            root_dir=data_path,
                            noise_file='%s/cifar10_noisy_labels_task1.json' % (data_path))

    trainloader, noisy_labels, clean_labels = loader.run('train')
    testloader = loader.run('test')
    
    classes = ('plane', 'car', 'bird', 'cat',
            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # BEGIN  Filtering only clean train samples!
    df_clean_indices = pd.read_csv('images-30epochs-2last_loss/clean_indices.csv')

    print(df_clean_indices.head())

    print(df_clean_indices.shape)

    train_indices = df_clean_indices['indices'].tolist()
    print("The number of training samples ", len(train_indices))

    clean_set = torch.utils.data.Subset(trainloader.dataset, train_indices)

    trainloader = torch.utils.data.DataLoader(dataset=clean_set, shuffle=True,
                                         batch_size=batch_size, drop_last=True)
    # End Filtering only clean train samples!

    # Defining the model
    model = ResNet(ResidualBlock, [2, 2, 2]).to(device)  # ResNet 18

    # todo: set up the optimizer and learning rate scheduler
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[3], gamma=0.1)

    # todo: set up the loss (empirical risk)
    criterion = nn.CrossEntropyLoss()


    # todo: start training
    print("######################Start Training###################################")
    epochs = 30
    for ep in range(epochs):
        model.train()

        correct = 0
        for batch_idx, (data, targets, _) in enumerate(trainloader):
            
            data, targets = data.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            # save accuracy:
            pred = outputs.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct += pred.eq(targets.view_as(pred)).sum().item()
            if batch_idx % 100 == 0:
                print(
                    'Train Epoch: {} [{}/{} ({:.0f}%)]\t Training Loss: {:.6f} Accuracy: {:.0f}%, Learning rate: {:.6f}'.format(
                        ep, batch_idx * len(data), len(trainloader.dataset),
                            100. * batch_idx / len(trainloader), loss.item(),
                            100. * correct / ((batch_idx + 1) * batch_size),
                        optimizer.param_groups[0]['lr']))
        scheduler.step()
        print()


    torch.save(model.state_dict(), "model_trained_before_cleaning-epoch30-again.pt")

    # In[16]:


    # todo: conduct testing
    print("######################Test###################################")
    model.eval()
    correct = 0
    with torch.no_grad():
        test_loss = 0
        correct = 0
        for batch_idx, (data, targets) in enumerate(testloader):
            data, targets = data.to(device), targets.to(device)
            outputs= model(data)
            test_loss += criterion(outputs,targets)
            pred = outputs.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct += pred.eq(targets.view_as(pred)).sum().item()

    test_loss /= len(testloader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(testloader.dataset),
        100. * correct / len(testloader.dataset)))


    
    

if __name__ == '__main__':
    main()
