import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import time
class Inception(nn.Module):
    def __init__(self,in_planes,kernel_1_x,kernel_3_in,kernel_3_x,kernel_5_in,kernel_5_x,pool_planes):
        super().__init__()
        # 1x1 conv branch
        self.b1 = nn.Sequential(
            nn.Conv2d(in_channels=in_planes,out_channels=kernel_1_x,kernel_size=1),
            nn.BatchNorm2d(kernel_1_x),
            nn.ReLU(True)
        )
        # 1x1 conv -> 3x3 conv branch
        self.b2 = nn.Sequential(
            nn.Conv2d(in_channels=in_planes,out_channels=kernel_3_in,kernel_size=1),
            nn.BatchNorm2d(kernel_3_in),
            nn.ReLU(True),
            nn.Conv2d(in_channels=kernel_3_in,out_channels=kernel_3_x,kernel_size=3,padding=1),
            nn.BatchNorm2d(kernel_3_x),
            nn.ReLU(True)
        )
        # 1x1 conv -> 5x5 conv branch
        self.b3 = nn.Sequential(
            nn.Conv2d(in_channels=in_planes, out_channels=kernel_5_in, kernel_size=1),
            nn.BatchNorm2d(kernel_5_in),
            nn.ReLU(True),
            nn.Conv2d(in_channels=kernel_5_in, out_channels=kernel_5_x, kernel_size=3, padding=1),
            nn.BatchNorm2d(kernel_5_x),
            nn.ReLU(True),
            nn.Conv2d(in_channels=kernel_5_x, out_channels=kernel_5_x, kernel_size=3, padding=1),
            nn.BatchNorm2d(kernel_5_x),
            nn.ReLU(True)
        )
        # 3x3 pool -> 1x1 conv branch
        self.b4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3,stride=1,padding=1),
            nn.Conv2d(in_channels=in_planes,out_channels=pool_planes,kernel_size=1),
            nn.BatchNorm2d(pool_planes),
            nn.ReLU(True)
        )

    def forward(self,t):
        y1 = self.b1(t)
        y2 = self.b2(t)
        y3 = self.b3(t)
        y4 = self.b4(t)

        return torch.cat([y1,y2,y3,y4],dim=1)



class GoogLeNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.pre_layers = nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=192,kernel_size=3,padding=1),
            nn.BatchNorm2d(192),
            nn.ReLU(True)
        )
        self.a3 = Inception(192, 64, 96, 128, 16, 32, 32)
        self.b3 = Inception(256, 128, 128, 192, 32, 96, 64)

        self.max_pool = nn.MaxPool2d(3, stride=2, padding=1)

        self.a4 = Inception(480, 192, 96, 208, 16, 48, 64)
        self.b4 = Inception(512, 160, 112, 224, 24, 64, 64)
        self.c4 = Inception(512, 128, 128, 256, 24, 64, 64)
        self.d4 = Inception(512, 112, 144, 288, 32, 64, 64)
        self.e4 = Inception(528, 256, 160, 320, 32, 128, 128)

        self.a5 = Inception(832, 256, 160, 320, 32, 128, 128)
        self.b5 = Inception(832, 384, 192, 384, 48, 128, 128)

        self.avgpool = nn.AvgPool2d(8, stride=1)
        self.linear = nn.Linear(1024, 10)
    def forward(self,x):
        x = self.pre_layers(x)
        x = self.a3(x)
        x = self.b3(x)
        x = self.max_pool(x)
        x = self.a4(x)
        x = self.b4(x)
        x = self.c4(x)
        x = self.d4(x)
        x = self.e4(x)
        x = self.max_pool(x)
        x = self.a5(x)
        x = self.b5(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x


lr = 0.001
epoch_num = 10
trainBatchSize = 100
testBatchSzie = 100
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
network = GoogLeNet().to(device)
network = GoogLeNet()
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
