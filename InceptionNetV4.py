import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import visdom
import os
import time
# 处理数据设备

GPU = torch.cuda.is_available()
device = torch.device('cuda:0' if GPU else 'cpu')
if GPU:
    print("使用GPU训练......")
else:
    print("使用CPU训练......")

# 初始化窗口
viz = visdom.Visdom()
viz.line([0.], [0.], win="train_loss", opts={
    "title": "train_loss"
})
viz.line([0.], [0.], win="val_loss", opts={
    "title": "val_loss"
})
viz.line([0.], [0.], win="val_acc", opts={
    "title": "acc"
})
# 训练数据超参数
train_parameters = {
    "learning_rate": 0.00005,
    "batch": 32,
    "epoch": 200
}
# 数据增强
train_db_transform = transforms.Compose([
    # 中心裁剪
    # transforms.CenterCrop(224),
    # 随机裁剪
    # transforms.RandomCrop(224),
    # 随机旋转角度 -45~45
    transforms.RandomRotation(45),
    # 随机水平翻转
    transforms.RandomHorizontalFlip(0.5),
    # 随机垂直翻转
    transforms.RandomVerticalFlip(0.5),
    # 转化成Tenosr 并且进行归一化
    transforms.ToTensor(),
    # 标准化
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
val_db_transforms = transforms.Compose([
    # 转化成Tenosr 并且进行归一化
    transforms.ToTensor(),
    # 标准化
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
test_db_transform = transforms.Compose([
    # 转化成Tenosr 并且进行归一化
    transforms.ToTensor(),
    # 标准化
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 获得数据
train_db = datasets.CIFAR10(root='./data', train=True, download=False, transform=train_db_transform)
# 划分训练集验证集 4:1
train_db, val_db = random_split(dataset=train_db, lengths=[40000, 10000])
test_db = datasets.CIFAR10(root='./data', train=False, download=False, transform=test_db_transform)


train_db_loader = DataLoader(train_db, shuffle=True, batch_size=train_parameters["batch"])
val_db_loader = DataLoader(val_db, shuffle=True, batch_size=train_parameters["batch"])
test_db_loader = DataLoader(test_db, shuffle=True, batch_size=train_parameters["batch"])

# 构建BNA

class Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding=0, stride=1):
        super(Conv, self).__init__()
        self.Conv = nn.Conv2d(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=kernel_size,
                              padding=padding,
                              stride=stride)
        self.BN = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.Conv(x)
        x = self.BN(x)
        output = F.relu(x)
        return output

# 构建 stem

class stem(nn.Module):
    def __init__(self):
        super(stem, self).__init__()
        self.Conv_1 = Conv(     in_channels=3,
                                out_channels=32,
                                kernel_size=(3, 3),
                                padding=1,
                                stride=2)

        self.Conv_2 = Conv(     in_channels=32,
                                out_channels=32,
                                kernel_size=(3, 3),
                                )

        self.Conv_3 = Conv( in_channels=32,
                            out_channels=64,
                            kernel_size=(3, 3),
                            padding=1)

        self.Max_Pool_4_1 = nn.MaxPool2d(kernel_size=(3, 3),
                                         stride=2,
                                         padding=1)

        self.Conv_4_2 = Conv(in_channels=64,
                                  out_channels=96,
                                  stride=2,
                                  padding=1,
                                  kernel_size=(3, 3))

        self.Conv_5_1 = Conv(in_channels=160,
                                  out_channels=64,
                                  kernel_size=(1, 1))

        self.Conv_5_2 = Conv(in_channels=64,
                                  out_channels=96,
                                  kernel_size=(3, 3),
                                  padding=1)

        self.Conv_6_1 = Conv(in_channels=192,
                                  out_channels=192,
                                  kernel_size=(3, 3),
                                  padding=1,
                                  stride=2)

        self.Max_Pool_6_2 = nn.MaxPool2d(kernel_size=(3, 3),
                                         padding=1,
                                         stride=2)

        self.Conv_7_1 = Conv(in_channels=160,
                                  out_channels=64,
                                  kernel_size=(1, 1))

        self.Conv_7_2 = Conv(in_channels=64,
                                  out_channels=64,
                                  kernel_size=(7, 1),
                                  padding=(3, 0))

        self.Conv_7_3 = Conv(in_channels=64,
                                  out_channels=64,
                                  kernel_size=(1, 7),
                                  padding=(0, 3))

        self.Conv_7_4 = Conv(in_channels=64,
                                  out_channels=96,
                                  kernel_size=(3, 3),
                                  padding=1)





    def forward(self, x):
        x = self.Conv_1(x)
        x = self.Conv_2(x)
        x = self.Conv_3(x)

        x1 = self.Max_Pool_4_1(x)
        x2 = self.Conv_4_2(x)
        x3 = torch.cat((x1, x2), dim=1)
        # x3 = F.leaky_relu(x3)

        x4 = self.Conv_5_1(x3)
        x4 = self.Conv_5_2(x4)

        x5 = self.Conv_7_1(x3)
        x5 = self.Conv_7_2(x5)
        x5 = self.Conv_7_3(x5)
        x5 = self.Conv_7_4(x5)

        x6 = torch.cat((x4, x5), dim=1)
        # x6 = F.leaky_relu(x6)
        x7 = self.Conv_6_1(x6)
        x8 = self.Max_Pool_6_2(x6)
        # output = F.leaky_relu(torch.cat((x7, x8), dim=1))
        output = torch.cat((x7, x8), dim=1)
        return output




class Inception_A(nn.Module):
    def __init__(self):
        super(Inception_A, self).__init__()
        self.Avg_pool_1_1 = nn.AvgPool2d(kernel_size=(1, 1),
                                         stride=1
        )

        self.Conv_1_2 = Conv(in_channels=384,
                                  out_channels=96,
                                  kernel_size=(1, 1),
                                  stride=1)

        self.Conv_2_1 = Conv(in_channels=384,
                                  out_channels=96,
                                  kernel_size=(1, 1),
                                  stride=1)

        self.Conv_3_1 = Conv(in_channels=384,
                                  out_channels=64,
                                  kernel_size=(1, 1),
                                  stride=1)

        self.Conv_4_1 = Conv(in_channels=384,
                                  out_channels=64,
                                  kernel_size=(1, 1),
                                  stride=1)



        self.Conv_3_2 = Conv(in_channels=64,
                                  out_channels=96,
                                  kernel_size=(3, 3),
                                  padding=1)

        self.Conv_4_2 = Conv(in_channels=64,
                                  out_channels=96,
                                  kernel_size=(3, 3),
                                  padding=1)

        self.Conv_4_3 = Conv(in_channels=96,
                                  out_channels=96,
                                  kernel_size=(3, 3),
                                  padding=1)

    def forward(self, x):
        x1 = self.Avg_pool_1_1(x)
        x1 = self.Conv_1_2(x1)

        x2 = self.Conv_2_1(x)

        x3 = self.Conv_3_1(x)
        x3 = self.Conv_3_2(x3)

        x4 = self.Conv_4_1(x)
        x4 = self.Conv_4_2(x4)
        x4 = self.Conv_4_3(x4)
        # output = F.leaky_relu(torch.cat((x1, x2, x3, x4), dim=1))
        output = torch.cat((x1, x2, x3, x4), dim=1)
        return output

class Reduction_A(nn.Module):
    def __init__(self):
        super(Reduction_A, self).__init__()
        self.Max_Pool_1_1 = nn.MaxPool2d(kernel_size=(3, 3),
                                         stride=2,
                                         padding=1)

        self.Conv_2_1 = Conv(in_channels=384,
                                  out_channels=384,
                                  kernel_size=(3, 3),
                                  stride=2,
                                  padding=1)

        self.Conv_3_1 = Conv(in_channels=384,
                                  out_channels=192,
                                  kernel_size=(1, 1),
                                  )

        self.Conv_3_2 = Conv(in_channels=192,
                                  out_channels=224,
                                  kernel_size=(3, 3),
                                  padding=1)

        self.Conv_3_3 = Conv(in_channels=224,
                                  out_channels=256,
                                  kernel_size=(3, 3),
                                  padding=1,
                                  stride=2
                                  )


    def forward(self, x):
        x1 = self.Max_Pool_1_1(x)
        x2 = self.Conv_2_1(x)

        x3 = self.Conv_3_1(x)
        x3 = self.Conv_3_2(x3)
        x3 = self.Conv_3_3(x3)
        # output = F.leaky_relu(torch.cat((x1, x2, x3), dim=1))
        output = torch.cat((x1, x2, x3), dim=1)
        return output

class Inception_B(nn.Module):
    def __init__(self):
        super(Inception_B, self).__init__()
        self.Avg_Pool_1_1 = nn.AvgPool2d(kernel_size=(1, 1),
                                         stride=1)
        self.Conv_1_2 = Conv(in_channels=1024,
                                  out_channels=128,
                                  kernel_size=(1, 1),
                                  stride=1)

        self.Conv_2_1 = Conv(in_channels=1024,
                                  out_channels=384,
                                  kernel_size=(1, 1),
                                  stride=1)

        self.Conv_3_1 = Conv(in_channels=1024,
                                  out_channels=192,
                                  kernel_size=(1, 1))

        self.Conv_3_2 = Conv(in_channels=192,
                                  out_channels=224,
                                  kernel_size=(1, 7),
                                  padding=(0, 3))

        self.Conv_3_3 = Conv(in_channels=224,
                                  out_channels=256,
                                  kernel_size=(1, 7),
                                  padding=(0, 3))

        self.Conv_4_1 = Conv(in_channels=1024,
                                  out_channels=192,
                                  kernel_size=(1, 1))

        self.Conv_4_2 = Conv(in_channels=192,
                                  out_channels=192,
                                  kernel_size=(1, 7),
                                  padding=(0, 3))

        self.Conv_4_3 = Conv(in_channels=192,
                                  out_channels=224,
                                  kernel_size=(7, 1),
                                  padding=(3, 0))

        self.Conv_4_4 = Conv(in_channels=224,
                                  out_channels=224,
                                  kernel_size=(1, 7),
                                  padding=(0, 3))

        self.Conv_4_5 = Conv(in_channels=224,
                                  out_channels=256,
                                  kernel_size=(7, 1),
                                  padding=(3, 0))

    def forward(self, x):
        x1 = self.Avg_Pool_1_1(x)
        x1 = self.Conv_1_2(x1)

        x2 = self.Conv_2_1(x)

        x3 = self.Conv_3_1(x)
        x3 = self.Conv_3_2(x3)
        x3 = self.Conv_3_3(x3)

        x4 = self.Conv_4_1(x)
        x4 = self.Conv_4_2(x4)
        x4 = self.Conv_4_3(x4)
        x4 = self.Conv_4_4(x4)
        x4 = self.Conv_4_5(x4)

        # output = F.leaky_relu(torch.cat((x1, x2, x3, x4), dim=1))
        output = torch.cat((x1, x2, x3, x4), dim=1)
        return output


class Reduction_B(nn.Module):
    def __init__(self):
        super(Reduction_B, self).__init__()
        self.Max_Pool_1_1 = nn.MaxPool2d(kernel_size=(3, 3),
                                         padding=1,
                                         stride=2)

        self.Conv_2_1 = Conv(in_channels=1024,
                                  out_channels=192,
                                  kernel_size=(1, 1)
                                  )

        self.Conv_2_2 = Conv(in_channels=192,
                                  out_channels=192,
                                  kernel_size=(3, 3),
                                  padding=1,
                                  stride=2)

        self.Conv_3_1 = Conv(in_channels=1024,
                                  out_channels=256,
                                  kernel_size=(1, 1))

        self.Conv_3_2 = Conv(in_channels=256,
                                  out_channels=256,
                                  kernel_size=(1, 7),
                                  padding=(0, 3))

        self.Conv_3_3 = Conv(in_channels=256,
                                  out_channels=320,
                                  kernel_size=(7, 1),
                                  padding=(3, 0))

        self.Conv_3_4 = Conv(in_channels=320,
                                  out_channels=320,
                                  kernel_size=(3, 3),
                                  padding=1,
                                  stride=2)

    def forward(self, x):
        x1 = self.Max_Pool_1_1(x)

        x2 = self.Conv_2_1(x)
        x2 = self.Conv_2_2(x2)

        x3 = self.Conv_3_1(x)
        x3 = self.Conv_3_2(x3)
        x3 = self.Conv_3_3(x3)
        x3 = self.Conv_3_4(x3)
        # output = F.leaky_relu(torch.cat((x1, x2, x3), dim=1))
        output = torch.cat((x1, x2, x3), dim=1)
        return output

class Inceptino_C(nn.Module):
    def __init__(self):
        super(Inceptino_C, self).__init__()
        self.Avg_Pool_1_1 = nn.AvgPool2d(kernel_size=(1, 1),
                                         )

        self.Conv_1_2 = Conv(in_channels=1536,
                                  out_channels=256,
                                  kernel_size=(1, 1))

        self.Conv_2_1 = Conv(in_channels=1536,
                                  out_channels=256,
                                  kernel_size=(1, 1))

        self.Conv_3_1 = Conv(in_channels=1536,
                                  out_channels=384,
                                  kernel_size=(1, 1))
        self.Conv_3_2_1 = Conv(in_channels=384,
                                    out_channels=256,
                                    kernel_size=(1, 3),
                                    padding=(0, 1))

        self.Conv_3_2_2 = Conv(in_channels=384,
                                    out_channels=256,
                                    kernel_size=(3, 1),
                                    padding=(1, 0))

        self.Conv_4_1 = Conv(in_channels=1536,
                                  out_channels=384,
                                  kernel_size=(1, 1))

        self.Conv_4_2 = Conv(in_channels=384,
                                  out_channels=448,
                                  kernel_size=(1, 3),
                                  padding=(0, 1))

        self.Conv_4_3 = Conv(in_channels=448,
                                  out_channels=512,
                                  kernel_size=(3, 1),
                                  padding=(1, 0))

        self.Conv_4_4_1 = Conv(in_channels=512,
                                    out_channels=256,
                                    kernel_size=(3, 1),
                                    padding=(1, 0))

        self.Conv_4_4_2 = Conv(in_channels=512,
                                    out_channels=256,
                                    kernel_size=(1, 3),
                                    padding=(0, 1))

    def forward(self, x):
        x1 = self.Avg_Pool_1_1(x)
        x1 = self.Conv_1_2(x1)

        x2 = self.Conv_2_1(x)

        x3 = self.Conv_3_1(x)
        x3_1 = self.Conv_3_2_1(x3)
        x3_2 = self.Conv_3_2_2(x3)

        x4 = self.Conv_4_1(x)
        x4 = self.Conv_4_2(x4)
        x4 = self.Conv_4_3(x4)
        x4_1 = self.Conv_4_4_1(x4)
        x4_2 = self.Conv_4_4_2(x4)

        # output = F.leaky_relu(torch.cat((x1, x2, x3_1, x3_2, x4_1, x4_2), dim=1))
        output = torch.cat((x1, x2, x3_1, x3_2, x4_1, x4_2), dim=1)
        return output


# 构建神经网络模型GoogleNet V4

class Googlenet_v4(nn.Module):
    def __init__(self):
        super(Googlenet_v4, self).__init__()
        self.stem = stem()
        self.In_A = Inception_A()
        self.Re_A = Reduction_A()
        self.In_B = Inception_B()
        self.Re_B = Reduction_B()
        self.In_C = Inceptino_C()
        self.Avg = nn.AvgPool2d(kernel_size=(1, 1))
        self.Flatten = nn.Flatten()
        self.Dropout = nn.Dropout(0.5)
        self.d1 = nn.Linear(1536, 200)
        self.d2 = nn.Linear(200, 10)

    def forward(self, x):
        x = self.stem(x)
        x = self.In_A(x)
        x = self.Re_A(x)
        x = self.In_B(x)
        x = self.Re_B(x)
        x = self.In_C(x)
        x = self.Avg(x)
        x = self.Flatten(x)
        x = self.d1(x)
        x = F.leaky_relu(x)
        x = self.d2(x)
        x = self.Dropout(x)
        output = F.softmax(x)
        return output

# 获取模型
model = Googlenet_v4()

# 判断是否存在断点
checkpoint_path = ""
RESUME = False
start_epoch = 0
if os.path.exists(checkpoint_path):
    print("断点路径:{0} 开始加载断点".format(checkpoint_path))
    RESUME = True
    checkpoint = torch.load(checkpoint_path, map_location=device)
    start_epoch = checkpoint["start_epoch"]
    optimizer = checkpoint["optimizer"]
    model.load_state_dict(checkpoint["weight"])
    print("加载断点完成")

else:
    print("不存在断点, 直接训练模型")
    if not os.path.exists("./checkpoint/"):
        print("在根目录下创建checkpoint文件夹")
        os.mkdir("./checkpoint/")

if not RESUME:
    optimizer = torch.optim.Adam(model.parameters(), lr=train_parameters["learning_rate"])
    # torch.optim.lr_scheduler.CosineAnnealingLR()
loss_function = torch.nn.CrossEntropyLoss()
model.to(device)

# 开始训练
global_step = 0
global_val_step = 0
best_acc = 0
best_loss = 0
save_path = ""
print("模型开始在 {0} 上训练".format(device))
for epoch in range(start_epoch, train_parameters["epoch"]):
    start_time = time.time()
    model.train()
    total_train_loss = 0
    for step, (x, y) in enumerate(train_db_loader):
        if GPU:
            x = x.cuda()
            y = y.cuda()
        global_step += 1
        pred = model(x)
        train_loss = loss_function(pred, y)
        total_train_loss += train_loss
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()
    # 打印输出平均loss
    avg_train_loss = total_train_loss / (step + 1)
    viz.line([avg_train_loss.item()], [epoch], win="train_loss", update="append")
    print("epoch:{0}  train_loss{1:.2f}".format(epoch, avg_train_loss))
    ###############################################
    # 保存模型参数
    ###############################################
    # if epoch % 5 == 0:
    #     checkpoint = {
    #         "optimizer": optimizer,
    #         "start_epoch": start_epoch,
    #         "weight": model.state_dict()
    #     }
    #     torch.save(checkpoint, "./checkpoint/_cifar_best_" + str(epoch) + ".pth")
    model.eval()
    with torch.no_grad():
        total_val_loss = 0
        total_acc = 0
        for step, (val_x, val_y) in enumerate(val_db_loader):
            correct = 0
            if GPU:
                val_x = val_x.cuda()
                val_y = val_y.cuda()
            global_val_step += 1
            pred = model(val_x)
            val_loss = loss_function(pred, val_y)
            total_val_loss += val_loss
            correct = torch.eq(torch.argmax(pred, dim=-1), val_y).float().sum()
            acc = 100 * correct / val_x.size(0)
            total_acc += acc

        avg_val_loss = total_val_loss / (step+1)
        avg_val_acc = total_acc / (step + 1)
        viz.line([avg_val_loss.item()], [epoch], win="val_loss", update="append")
        viz.line([avg_val_acc.item()], [epoch], win="val_acc", update="append")
        print("epoch:{0}  acc:{1}   val_loss:{2}".format(epoch, avg_val_acc, avg_val_loss))
        if avg_val_acc > best_acc and epoch % 10 == 0:
            best_acc = avg_val_acc
            best_loss = avg_val_loss
            checkpoint = {
                "optimizer": optimizer,
                "start_epoch": epoch,
                "weight": model.state_dict()
            }
            save_path = "./checkpoint/_cifar_best_" + str(epoch) + ".pth"
            torch.save(checkpoint, save_path)
        end_time = time.time()
        cost_time = end_time - start_time
        print("cost_time:{0}s".format(cost_time))
print("模型训练结束, 最佳acc:{0:.2f}, val_loss:{1:.2f} 模型参数保存在路径:{2}".format(best_acc, best_loss, save_path))
# 测试代码正确性
