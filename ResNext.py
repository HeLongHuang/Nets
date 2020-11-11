import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import time

CARDINALITY = 32
DEPTH = 4
BASEWIDTH = 64


class ResNextBottleNeckC(nn.Module):

    def __init__(self, in_channels, out_channels, stride):
        super().__init__()

        C = CARDINALITY #How many groups a feature map was splitted into

        #"""We note that the input/output width of the template is fixed as
        #256-d (Fig. 3), We note that the input/output width of the template
        #is fixed as 256-d (Fig. 3), and all widths are dou- bled each time
        #when the feature map is subsampled (see Table 1)."""
        D = int(DEPTH * out_channels / BASEWIDTH) #number of channels per group
        self.split_transforms = nn.Sequential(
            nn.Conv2d(in_channels, C * D, kernel_size=1, groups=C, bias=False),
            nn.BatchNorm2d(C * D),
            nn.ReLU(inplace=True),
            nn.Conv2d(C * D, C * D, kernel_size=3, stride=stride, groups=C, padding=1, bias=False),
            nn.BatchNorm2d(C * D),
            nn.ReLU(inplace=True),
            nn.Conv2d(C * D, out_channels * 4, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels * 4),
        )

        self.shortcut = nn.Sequential()

        if stride != 1 or in_channels != out_channels * 4:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * 4, stride=stride, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels * 4)
            )

    def forward(self, x):
        return F.relu(self.split_transforms(x) + self.shortcut(x))

class ResNext(nn.Module):

    def __init__(self, block, num_blocks, class_names=100):
        super().__init__()
        self.in_channels = 64

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.conv2 = self._make_layer(block, num_blocks[0], 64, 1)
        self.conv3 = self._make_layer(block, num_blocks[1], 128, 2)
        self.conv4 = self._make_layer(block, num_blocks[2], 256, 2)
        self.conv5 = self._make_layer(block, num_blocks[3], 512, 2)
        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * 4, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.avg(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def _make_layer(self, block, num_block, out_channels, stride):
        """Building resnext block
        Args:
            block: block type(default resnext bottleneck c)
            num_block: number of blocks per layer
            out_channels: output channels per block
            stride: block stride
        Returns:
            a resnext layer
        """
        strides = [stride] + [1] * (num_block - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * 4

        return nn.Sequential(*layers)

def resnext50():
    """ return a resnext50(c32x4d) network
    """
    return ResNext(ResNextBottleNeckC, [3, 4, 6, 3])

def resnext101():
    """ return a resnext101(c32x4d) network
    """
    return ResNext(ResNextBottleNeckC, [3, 4, 23, 3])

def resnext152():
    """ return a resnext101(c32x4d) network
    """
    return ResNext(ResNextBottleNeckC, [3, 4, 36, 3])



# x = torch.rand(15,3,28,28)
# network = resnext50()
# with SummaryWriter(comment='Net1')as w:
#     w.add_graph(network,(x,))


lr = 0.001
epoch_num = 10
trainBatchSize = 200
testBatchSzie = 200
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
network = resnext50().to(device)
print(network)
optimizer = optim.Adam(network.parameters(),lr = lr)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[75, 150], gamma=0.5)
criterion = nn.CrossEntropyLoss()


train_transform = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.ToTensor()])
test_transform = transforms.Compose([transforms.ToTensor()])
train_set = torchvision.datasets.CIFAR10(root='./Cifar-10', train=True, download=True, transform=train_transform)
train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=trainBatchSize, shuffle=True)
test_set = torchvision.datasets.CIFAR10(root='./Cifar-10', train=False, download=True, transform=test_transform)
test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=testBatchSzie, shuffle=False)


def train():
    print("******************training******************")
    for epoch in range(epoch_num):
        total_correct = 0
        total_loss = 0
        total_sample = 0
        for batch_num,(data,target) in enumerate(train_loader):
            t1 = time.time()
            data = data.to(device)
            target = target.to(device)
            output = network(data)
            # print("output",output)
            step_loss = criterion(output,target)
            optimizer.zero_grad()
            step_loss.backward()
            # step_loss.to(device)
            optimizer.step()
            _, prediction = torch.max(output, 1)
            # print("prediction", prediction)
            step_correct = (prediction == target).sum()
            # print("step_correct",step_correct)
            total_correct += step_correct
            total_sample += target.size(0)
            total_loss += step_loss
            t2 = time.time()
            # print("epoch:",epoch,"step:",batch_num,"step_loss:",step_loss.item(),"step_acc:",(step_correct / trainBatchSize).item(),"time:",t2 - t1)
            if batch_num % 10 == 0:
              print("epoch:{0:>3} =======> step:{1:>3}  step_loss:{2:>20}  step_acc:{3:>20}  time:{4:>20}".format(epoch,batch_num,step_loss.item(),(step_correct / trainBatchSize).item(),t2 - t1))
        print("********************************************************************")
        print("epoch:",epoch,"loss:",total_loss,"acc:",total_correct / total_sample)
        print("********************************************************************")
    torch.save(network, '/googleNet_model.pkl')


def test():
    print("******************test******************")
    network.eval()
    test_loss = 0
    test_correct = 0
    total = 0
    with torch.no_grad():
        for batch_num,(data,target) in enumerate(test_loader):
            data = data.to(device)
            target = target.to(device)
            output = network(data)
            loss = criterion(output,target)
            test_loss += loss.item()
            _, prediction = torch.max(output, 1)
            total += target.size(0)
            current_correct = (prediction == target).sum()
            # print("currentcorrect",current_correct)
            test_correct += current_correct
            # print("test_batch_num:",batch_num,"test_batch_correct:",test_correct.item(),"test_batch_acc:",(current_correct / testBatchSzie).item())
            # print("epoch:{0:>3} =======> step:{1:>3}  step_loss:{2:>20}  step_acc:{3:>20}  time:{4:>20}".format(epoch,batch_num,step_loss.item(),(step_correct / trainBatchSize).item(),t2 - t1))
            print("test_batch_num:{0:>3}  test_batch_correct:{1:>5}  test_batch_acc:{2:>20}".format(batch_num,test_correct.item(),(current_correct / testBatchSzie).item()))


if __name__ == '__main__':
    print(network)
    train()
    test()
