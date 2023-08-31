import os.path
import torch
from torch import nn
from torch import optim
from torchvision import models
from utils.appledata import CustomDataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import time
from torch.utils.data import Dataset, ConcatDataset
from mymodel import se_resnext50_32x4d
import cv2
import numpy as np
import torchvision.transforms as transforms
from PIL import Image
from utils.applePseudoData import pseudoDataset

from lrs_scheduler import WarmRestart, warm_restart
from sklearn.model_selection import KFold

k_folds = 5
kf = KFold(n_splits=k_folds, shuffle=True)

train_transform = transforms.Compose([
    transforms.Resize((550, 550)),
    transforms.RandomChoice([
        transforms.ColorJitter(brightness=0.1),
        transforms.ColorJitter(contrast=0.1),
    ]),
    transforms.RandomApply([transforms.Lambda(lambda img: Image.fromarray(cv2.medianBlur(np.array(img), 3))),
                            transforms.Lambda(
                                lambda img: Image.fromarray(cv2.GaussianBlur(np.array(img), (3, 3), 0))),
                            transforms.Lambda(lambda img: Image.fromarray(cv2.blur(np.array(img), (3, 3)))),
                            ], p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomAffine(
        degrees=20,
        translate=(0.2, 0.2),
        scale=(0.8, 1.2),
        interpolation=transforms.InterpolationMode.BILINEAR,
        fill=0,
    ),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),

])
val_transform = transforms.Compose([
    transforms.Resize((550, 550)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),

])


class TransformedDataset(Dataset):
    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.transform = transform

    def __getitem__(self, idx):
        x, y, z = self.dataset[idx]
        x = self.transform(x)
        return x, y, z

    def __len__(self):
        return len(self.dataset)


class ICLearner():

    def __init__(self, scheduler_step, gamma, classes, batch_size, lr, ic_model, path):
        self.ic_model = ic_model
        self.scheduler_step = scheduler_step
        self.gamma = gamma
        self.classes = classes
        self.batch_size = batch_size
        self.lr = lr
        self.path = path

        self.model = se_resnext50_32x4d()
        # 设置损失计算方法为交叉熵损失
        self.criterion = nn.CrossEntropyLoss()

        # 初始化优化器
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.0001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
        self.scheduler = WarmRestart(self.optimizer, T_max=10, T_mult=1, eta_min=1e-5)
        # 设初始化数据集
        self.dataset = CustomDataset(self.path)  # 数据集路径
        self.pseudo_dataset = pseudoDataset(self.path)
        print('len dataset is %d' % len(self.dataset))

        # 设置cuda
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # self.device = torch.device("cpu")
        print(self.device)
        self.model = self.model.to(self.device)
        # 设置损失、准确率 后面画图用
        self.loss_plt = []
        self.acc_plt = []
        self.loss_best = 0

    # 不要计算梯度，否则测试的时候内存会爆掉，我们确定不会调用Tensor.backward()函数
    def labeling(self, pseudo_dataset, name):
        self.load_model(name)
        pseudo_dataset = TransformedDataset(pseudo_dataset, val_transform)
        pseudo_dataloader = DataLoader(pseudo_dataset, batch_size=self.batch_size, shuffle=True, num_workers=0)
        self.model.eval()
        with torch.no_grad():
            for images, labels, index in pseudo_dataloader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                _, predicted = torch.max(outputs.data, 1)
                for i in range(len(index)):
                    self.pseudo_dataset.label[index[i]] = predicted[i].item()

    def save_model(self, name):
        dir_path = 'models/apple_model/' + name
        torch.save(self.model.state_dict(), dir_path)

    def load_model(self, name):
        dir_path = 'models/apple_model/' + name
        self.model.load_state_dict(torch.load(dir_path))

    def eval(self, images, label):
        if self.ic_model == 'inception_v3':
            if self.model.training:
                output, aux_output = self.model(images)
                loss1 = self.criterion(output, label)
                loss2 = self.criterion(aux_output, label)
                loss = loss1 + 0.4 * loss2
            else:
                output = self.model(images)
                loss = self.criterion(output, label)
        else:
            output = self.model(images)
            loss = self.criterion(output, label)

        _, pred = torch.max(output, 1)  # 前一个是输出的最大概率值是多少，后一个是最大概率对应的标签即类别
        return loss, pred

    def train(self, train_dataset, val_dataset, num_epochs, model_name):
        # 数据transform
        train_dataset = TransformedDataset(train_dataset, train_transform)
        val_dataset = TransformedDataset(val_dataset, val_transform)

        # 创建数据加载器
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, num_workers=0)

        best_acc = 0
        print('开始训练')
        for epoch in range(num_epochs):
            train_loss = 0
            train_correct = 0
            train_total = 0
            # 设置模型训练状态
            self.model.train()
            # count = 0
            for images, labels, _ in train_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()  # 梯度清零
                loss, predicted = self.eval(images, labels)  # 预测结果
                loss.backward()  # 计算梯度值
                self.optimizer.step()  # 更新权重

                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()
                train_loss += loss.item()
                # count += 1
                # print('batch{} over'.format(count))

            # 验证模型
            print('开始验证')
            self.model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            with torch.no_grad():
                for images, labels, _ in val_loader:
                    images, labels = images.to(self.device), labels.to(self.device)
                    loss, predicted = self.eval(images, labels)
                    val_loss += loss.item()
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()
            self.scheduler.step()

            # 打印训练和验证的损失和准确率
            # print('len of train_loader is {},len of val_loader is {}'.format(len(train_loader), len(val_loader)))
            # print('len of train_total is {},len of val_total is {}'.format(train_total, val_total))
            # print('len of train_correct is {},len of val_correct is {}'.format(train_correct, val_correct))
            train_loss /= len(train_loader)
            train_accuracy = 100.0 * train_correct / train_total
            val_loss /= len(val_loader)
            val_accuracy = 100.0 * val_correct / val_total
            print(
                f"Epoch {epoch + 1}/{num_epochs}: Train Loss: {train_loss:.4f} | Train Acc: {train_accuracy:.2f}% | Val Loss: {val_loss:.4f} | Val Acc: {val_accuracy:.2f}%")
            if val_accuracy > best_acc:
                best_acc = val_accuracy
                self.save_model(model_name)
            self.loss_plt.append(train_loss)
            self.acc_plt.append(val_accuracy)
        print('---------best acc is:', best_acc)

    def new_train(self, num_epochs):
        for fold, (train_indices, val_indices) in enumerate(kf.split(self.dataset)):
            print(f"Fold {fold + 1}/{k_folds}")
            model_name = '{}fold.pth'.format(fold + 1)
            # 根据折数拆分训练集和验证集
            train_dataset = torch.utils.data.Subset(self.dataset, train_indices)
            val_dataset = torch.utils.data.Subset(self.dataset, val_indices)
            # 训练模型
            self.train(train_dataset, val_dataset, num_epochs, model_name)
            self.plt_loss_acc()

            # 打伪标签
            self.labeling(self.pseudo_dataset, model_name)

            # 混合数据集
            mix_dataset = ConcatDataset([train_dataset, self.pseudo_dataset])
            # mix_dataloader = DataLoader(mix_dataset, batch_size=self.batch_size, shuffle=True, drop_last=True)

            # 利用含有伪标签的混合数据集进行训练
            model_name = '{}fold_pseudo.pth'.format(fold + 1)
            self.train(mix_dataset, val_dataset, num_epochs, model_name)
            self.plt_loss_acc()

    @torch.no_grad()
    def eval2(self, dataloader, name):
        self.load_model(name)
        self.model.eval()

        running_correct = 0
        for images, label, _ in dataloader:
            images = images.to(self.device)
            label = label.to(self.device)
            output = self.model(images)
            _, pred = torch.max(output, 1)
            # noinspection PyTypeChecker
            running_correct += torch.sum(pred == label)

        return running_correct / float(len(dataloader.dataset))

    # 绘制损失、准确率曲线
    def plt_loss_acc(self):
        # 判断result是否存在，否则创建
        if not os.path.exists('./results'):
            os.makedirs("results")
        # 获取当前时间
        current_time = time.strftime("%Y%m%d-%H%M%S")
        fig = plt.figure()
        plt.plot(self.loss_plt, label='loss')
        plt.ylabel('loss')
        plt.xlabel('epoche')
        plt.savefig('./results/' + current_time + '_loos_apple.jpg')
        plt.close(fig)

        fig = plt.figure()
        plt.plot(self.acc_plt, label='acc')
        plt.ylabel('acc')
        plt.xlabel('epoche')
        plt.savefig('./results/' + current_time + '_acc_apple.jpg')
        plt.close(fig)
        # 绘制完毕，清空
        self.loss_plt = []
        self.acc_plt = []
        pass

    def train11(self, n_epoches, dataloader, name):
        best_acc = 0
        for epoch in range(n_epoches):
            print('Epoch {}/{}  lr:{} ...'.format(epoch + 1, n_epoches,
                                                  self.optimizer.state_dict()['param_groups'][0]['lr']), end='')
            self.model.train()
            # 初始化损失、准确率
            running_loss = 0.0
            running_correct = 0
            size_batch = dataloader.batch_size
            print(size_batch)
            for images, label, _ in dataloader:
                # 放置到cuda上
                images = images.to(self.device)
                label = label.to(self.device)

                # 梯度清零
                self.optimizer.zero_grad()

                # 预测结果
                loss, pred = self.eval(images, label)

                # 计算梯度值
                loss.backward()

                # 更新权重
                self.optimizer.step()
                # if epoch + 1 < n_epoches - 4:
                #     self.scheduler = warm_restart(self.scheduler, T_mult=2)

                # 计算累积损失
                running_loss += loss.item() * size_batch
                # noinspection PyTypeChecker
                # 上面是去掉警告用的，下面是两个tensor做运算，结果一般是tensor，但结果是bool，出现警告，我们期望的就是bool
                running_correct += torch.sum(pred == label)

            # 每到一定步数(scheduler_step)，降低学习率(gamma)
            # self.lr_scheduler.step()
            self.scheduler.step()

            # 每轮结束，记录损失，打印当前损失
            running_loss = running_loss / float(len(dataloader.dataset))
            running_correct = running_correct / float(len(dataloader.dataset))
            self.loss_plt.append(running_loss)
            self.acc_plt.append(running_correct.cpu())
            print('\nTrain Loss: {:.4f}\tTrain Accuracy: {:.4f}\n'.format(running_loss, running_correct))

            if epoch + 1 < n_epoches - 4:
                self.scheduler = warm_restart(self.scheduler, T_mult=2)

            # 保存准确率最好的模型
            if running_correct > best_acc:
                best_acc = running_correct
                self.save_model(name)
                self.loss_best = running_loss
            # 保存最后一次训练的模型
            if epoch + 1 == n_epoches:
                self.save_model('final_' + name)
