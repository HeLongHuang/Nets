import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim


# cuda
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 加载数据集 (训练集和测试集)
trainset = torchvision.datasets.CIFAR10(
    root='./Cifar-10',
    train=True,
    download=True,
    transform=transforms.ToTensor())
trainloader = torch.utils.data.DataLoader(trainset, batch_size=100, shuffle=True)
testset = torchvision.datasets.CIFAR10(
    root='./Cifar-10',
    train=False,
    download=True,
    transform=transforms.ToTensor())
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False)

# 定义神经网络
class Net(nn.Module):  # 训练 ALexNet
    '''
    三层卷积，三层全连接  (应该是5层卷积，由于图片是 32 * 32，且为了效率，这里设成了 3 层)
    '''
    def __init__(self):
        super(Net, self).__init__()
        # 五个卷积层
        self.conv1 = nn.Sequential(  # 输入 32 * 32 * 3
            nn.Conv2d(in_channels=3, out_channels=6, kernel_size=3, stride=1, padding=1),   # (32-3+2)/1+1 = 32
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0)  # (32-2)/2+1 = 16
        )
        self.conv2 = nn.Sequential(  # 输入 16 * 16 * 6
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=3, stride=1, padding=1),  # (16-3+2)/1+1 = 16
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0)  # (16-2)/2+1 = 8
        )
        self.conv3 = nn.Sequential(  # 输入 8 * 8 * 16
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1), # (8-3+2)/1+1 = 8
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0)  # (8-2)/2+1 = 4
        )
        self.conv4 = nn.Sequential(  # 输入 4 * 4 * 64
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1), # (4-3+2)/1+1 = 4
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0)  # (4-2)/2+1 = 2
        )
        self.conv5 = nn.Sequential(  # 输入 2 * 2 * 128
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),# (2-3+2)/1+1 = 2
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0)  # (2-2)/2+1 = 1
        )                            # 最后一层卷积层，输出 1 * 1 * 128
        # 全连接层
        self.dense = nn.Sequential(
            nn.Linear(128, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 10)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = x.view(-1, 128)
        x = self.dense(x)
        return x

network = Net().to(device)
loss_func = nn.CrossEntropyLoss()  # 交叉熵损失
optimizer = optim.SGD(network.parameters(), lr=0.01, momentum=0.9)  # 随机梯度下降


def get_num_correct(preds,labels):
    return preds.argmax(dim = 1).eq(labels).sum().item()

for epoch in range(10):
    total_loss = 0
    total_correct = 0


    for batch in trainloader:
        images,labels = batch
        preds = network(images)
        loss = loss_func(preds,labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss
        total_correct += get_num_correct(preds,labels)
    print("epoch", epoch, "total_correct", total_correct, "loss", total_loss)


