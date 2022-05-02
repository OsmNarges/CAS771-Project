import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import matplotlib.pyplot as plt
import numpy as np

# todo: prepare the training data and test data
gpuid = '1'

os.environ['CUDA_VISIBLE_DEVICES'] = gpuid
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('We are using', device)

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

batch_size = 64

train_set = torchvision.datasets.CIFAR10(root='./data', train=True,
                                         download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,
                                           shuffle=True, num_workers=2)

test_set = torchvision.datasets.CIFAR10(root='./data', train=False,
                                        download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size,
                                          shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# def imshow(img):
#     img = img / 2 + 0.5     # unnormalize
#     npimg = img.numpy()
#     plt.imshow(np.transpose(npimg, (1, 2, 0)))
#     plt.show()
#
# dataiter = iter(train_loader)
# images, labels = dataiter.next()
#
# # print images
# imshow(torchvision.utils.make_grid(images))
# print('GroundTruth: ', ' '.join(f'{classes[labels[j]]:5s}' for j in range(64)))
# exit(0)

# todo: prepare the neural network architecture
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

model = Net().to(device)

# test the net
# data = torch.randn(1, 3, 32, 32).to(device)
# print(model(data).shape)

# todo: set up the optimizer and learning rate scheduler
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[3], gamma=0.1)

# todo: set up the loss (empirical risk)
criterion = nn.CrossEntropyLoss()

# or self define
# class DemoLoss(nn.Module):
#     def __init__(self, num_classes=10):
#         super(DemoLoss, self).__init__()
#         self.num_classes = num_classes
#
#     def forward(self, logits, labels):
#         pred = F.softmax(logits, dim=1)
#         label_one_hot = F.one_hot(labels, self.num_classes).float().to(pred.device)
#         ce = (-1 * torch.sum(label_one_hot * torch.log(pred), dim=1))
#         return ce.mean()
# criterion = DemoLoss(10)


# todo: start training
print("######################Start Training###################################")
epochs = 5
for ep in range(epochs):
    model.train()

    correct = 0
    for batch_idx, (data, targets) in enumerate(train_loader):
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
                    ep, batch_idx * len(data), len(train_loader.dataset),
                        100. * batch_idx / len(train_loader), loss.item(),
                        100. * correct / ((batch_idx + 1) * batch_size),
                    optimizer.param_groups[0]['lr']))
    scheduler.step()
    print()

# todo: conduct testing
print("######################Test###################################")
model.eval()
correct = 0
with torch.no_grad():
    test_loss = 0
    correct = 0
    for batch_idx, (data, targets) in enumerate(test_loader):
        data, targets = data.to(device), targets.to(device)
        outputs= model(data)
        test_loss += criterion(outputs,targets)
        pred = outputs.max(1, keepdim=True)[1]  # get the index of the max log-probability
        correct += pred.eq(targets.view_as(pred)).sum().item()

test_loss /= len(test_loader.dataset)
print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
    test_loss, correct, len(test_loader.dataset),
    100. * correct / len(test_loader.dataset)))
