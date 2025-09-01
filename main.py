import torch
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from net import CIFAR10Net,CNNNet,ResNet18,ViT
from utils import train_model,test_model,write2csv



EPOCHS = 100  # 训练轮数
BATCH_SIZE = 128  # 每批处理的数据量
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 使用GPU或者CPU
LEARNING_RATE = 0.0001
Transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),  # 随机裁剪
    transforms.RandomHorizontalFlip(),     # 随机水平翻转
    transforms.ToTensor(),  # numpy -> Tensor
    transforms.Normalize((0.4914,0.4822,0.4465), (0.2023,0.1994,0.2010))  # 正则化，降低模型的复杂度
])
# 下载数据集
train_set = datasets.CIFAR10(root='./CIFAR10/', train=True, transform=Transform, download=True)
test_set = datasets.CIFAR10(root='./CIFAR10/', train=False, transform=Transform, download=True)

# type(train_set.data): <class 'numpy.ndarray'>,train_set.data.shape: (50000, 32, 32, 3)
print("1type(train_set.data): {},train_set.data.shape: {}".format(type(train_set.data),train_set.data.shape))
# type(train_set.targets): <class 'list'>,len(train_set.targets): 50000
print("2type(train_set.targets): {},len(train_set.targets): {}".format(type(train_set.targets),len(train_set.targets)))
img, label = train_set[0]
print("2.5type(img), img.shape:",type(img), img.shape,label)    # <class 'torch.Tensor'> torch.Size([3, 32, 32])

# type(test_set.data): <class 'numpy.ndarray'>,test_set.data.shape: (10000, 32, 32, 3)
print("3type(test_set.data): {},test_set.data.shape: {}".format(type(test_set.data),test_set.data.shape))
# type(test_set.targets: <class 'list'>,len(test_set.targets): 10000
print("4type(test_set.targets: {},len(test_set.targets): {}".format(type(test_set.targets),len(test_set.targets)))



# load data
# train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=12,pin_memory=True)
# shuffle打乱数据,pin_memory加载到GPU中
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True,num_workers=0)

# test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=12,pin_memory=True)
# shuffle打乱数据
test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=True,num_workers=0)



# classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')



# model = CIFAR10Net().to(DEVICE)
# model = CNNNet().to(DEVICE)
# model = ResNet18().to(DEVICE)
model = ViT(img_size=32, patch_size=4, emb_size=128, depth=6).to(DEVICE)

optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9)
# optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

for epoch in range(1,EPOCHS+1):
    TrainLoss, TrainAccuracy=train_model(model, DEVICE, train_loader, optimizer,epoch)
    TestLoss, TestAccuracy=test_model(model, DEVICE, test_loader)
    write2csv(epoch,TrainLoss, TrainAccuracy,TestLoss, TestAccuracy)
    scheduler.step()

