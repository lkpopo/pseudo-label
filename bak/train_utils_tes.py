import os
import math
import numpy as np
import torchvision.transforms as transforms
import pandas as pd
import pretrainedmodels
from torch import nn
from torch.utils.data import Dataset
from torchvision import models
from PIL import Image
from torch.optim import lr_scheduler
import torch
import cv2


class CrossEntropyLossOneHot(nn.Module):
    def __init__(self):
        super(CrossEntropyLossOneHot, self).__init__()
        self.log_softmax = nn.LogSoftmax(dim=-1)

    def forward(self, preds, labels):
        return torch.mean(torch.sum(-labels * self.log_softmax(preds), -1))


def l2_norm(input, axis=1):
    norm = torch.norm(input, 2, axis, True)
    output = torch.div(input, norm)
    return output


class BinaryHead(nn.Module):
    def __init__(self, num_class=4, emb_size=2048, s=16.0):
        super(BinaryHead, self).__init__()
        self.s = s
        self.fc = nn.Sequential(nn.Linear(emb_size, num_class))

    def forward(self, fea):
        fea = l2_norm(fea)
        logit = self.fc(fea) * self.s
        return logit


class se_resnext50_32x4d(nn.Module):
    def __init__(self, pre, num_class):
        super(se_resnext50_32x4d, self).__init__()

        if pre:
            s = "imagenet"
        else:
            s = None
        self.model_ft = nn.Sequential(
            *list(pretrainedmodels.__dict__["se_resnext50_32x4d"](num_classes=1000, pretrained=s).children())[
             :-2
             ]
        )
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.model_ft.last_linear = None
        self.fea_bn = nn.BatchNorm1d(2048)
        self.fea_bn.bias.requires_grad_(False)
        self.binary_head = BinaryHead(num_class, emb_size=2048, s=1)
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        img_feature = self.model_ft(x)
        img_feature = self.avg_pool(img_feature)
        img_feature = img_feature.view(img_feature.size(0), -1)
        fea = self.fea_bn(img_feature)
        # fea = self.dropout(fea)
        output = self.binary_head(fea)

        return output


def generate_transform(dataset_name):
    if dataset_name == 'apple':
        train_transform = transforms.Compose([
            transforms.Resize((550, 550)),
            transforms.RandomChoice([
                transforms.ColorJitter(brightness=0.1),
                transforms.ColorJitter(contrast=0.1),
            ]),
            transforms.RandomApply([transforms.GaussianBlur(3, 0.2)], p=0.5),

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
    else:
        train_transform = transforms.Compose([transforms.Resize((600, 600)),
                                              transforms.RandomResizedCrop(350),  # 随即裁剪到224*224像素
                                              transforms.RandomHorizontalFlip(),  # 水平方向水机反转
                                              transforms.ToTensor(),  # 转换成tensor
                                              transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])  # 标准化

        val_transform = transforms.Compose([transforms.Resize((600, 600)),  # cannot 299, must (299, 299)，
                                            transforms.ToTensor(),
                                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    return {"train_transforms": train_transform, "val_transforms": val_transform}


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


def initialize_model(model_name, num_classes, pretrain):
    if model_name == 'inception_v3':
        model = models.inception_v3(
            weights='Inception_V3_Weights.IMAGENET1K_V1') if pretrain else models.inception_v3()
        num_ftrs = model.AuxLogits.fc.in_features
        model.AuxLogits.fc = nn.Linear(num_ftrs, 20)
        num_ftrs = model.fc.in_features  # 得到特征的个数
        model.fc = nn.Linear(num_ftrs, num_classes)

    elif model_name == 'vgg16':
        model = models.vgg16_bn(weights='VGG16_BN_Weights.IMAGENET1K_V1') if pretrain else models.vgg16_bn()
        num_ftrs = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(num_ftrs, num_classes)

    elif model_name == 'vgg19':
        model = models.vgg19_bn(weights='VGG19_BN_Weights.IMAGENET1K_V1') if pretrain else models.vgg19_bn()

        num_ftrs = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(num_ftrs, num_classes)

    elif model_name == 'densenet161':
        model = models.densenet161(
            weights='DenseNet161_Weights.IMAGENET1K_V1') if pretrain else models.densenet161()
        num_ftrs = model.classifier.in_features
        model.classifier = nn.Linear(num_ftrs, num_classes)

    elif model_name == 'resnet101':
        model = models.resnet101(weights='ResNet101_Weights.IMAGENET1K_V1') if pretrain else models.resnet101()
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)

    elif model_name == 'resnet50':
        model = models.resnext50_32x4d(
            weights='ResNeXt50_32X4D_Weights.IMAGENET1K_V1') if pretrain else models.resnext50_32x4d()
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)

    elif model_name == 'mobilenet':
        model = models.mobilenet_v3_large(
            weight='MobileNet_V3_Large_Weights.IMAGENET1K_V1') if pretrain else models.mobilenet_v3_large()
        num_ftrs = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(num_ftrs, num_classes)

    elif model_name == 'se_resnet50':
        # model = pretrainedmodels.se_resnext50_32x4d(
        #     pretrained='imagenet', ) if pretrain else pretrainedmodels.se_resnext50_32x4d(
        #     pretrained=None)
        # model = nn.Sequential(*list(model.children())[:-1])
        # model = nn.Sequential(
        #     model,
        #     nn.AdaptiveAvgPool2d((1, 1)),
        #     nn.Flatten(),
        #     nn.BatchNorm1d(2048),
        #     nn.Linear(2048, num_classes)
        # )
        model = se_resnext50_32x4d(pretrain, num_classes)

    else:
        raise ValueError('Template model with model_name {} not defined'.format(model_name))

    return model


class PlantDataset(Dataset):

    def __init__(self, data, dataset_name, transforms=None):
        self.IMAGE_PATH = f'data/{dataset_name}'
        self.data = data
        self.transforms = transforms
        self.labels = self.data['label'].tolist()

    def __getitem__(self, idx):
        image_id = self.data.loc[idx, 'image_id']

        # image = Image.open(os.path.join(self.IMAGE_PATH, f"{image_id}.png")).convert('RGB')
        image = cv2.cvtColor(
            cv2.imread(os.path.join(self.IMAGE_PATH, f"{image_id}.png")), cv2.COLOR_BGR2RGB
        )
        if self.transforms:
            image = self.transforms(image)

        image = image.transpose(2, 0, 1)
        image = np.array(image).astype(np.float32)
        return image, self.labels[idx], idx

    def __len__(self):
        return len(self.data)


def load_data(data_name, frac=1):
    data = pd.read_csv(f"data/{data_name}.csv")
    if frac < 1:
        data = data.sample(frac=frac).reset_index(drop=True)
    plant_dataset = PlantDataset(data, data_name, transforms=None)
    return plant_dataset


class WarmRestart(lr_scheduler.CosineAnnealingLR):
    """This class implements Stochastic Gradient Descent with Warm Restarts(SGDR): https://arxiv.org/abs/1608.03983.

    Set the learning rate of each parameter group using a cosine annealing schedule,
    When last_epoch=-1, sets initial lr as lr.
    This can't support scheduler.step(epoch). please keep epoch=None.
    """

    def __init__(self, optimizer, T_max=10, T_mult=2, eta_min=0, last_epoch=-1):
        """implements SGDR

        Parameters:
        ----------
        T_max : int
            Maximum number of epochs.
        T_mult : int
            Multiplicative factor of T_max.
        eta_min : int
            Minimum learning rate. Default: 0.
        last_epoch : int
            The index of last epoch. Default: -1.
        """
        self.T_mult = T_mult
        super().__init__(optimizer, T_max, eta_min, last_epoch)

    def get_lr(self):
        if self.last_epoch == self.T_max:
            self.last_epoch = 0
            self.T_max *= self.T_mult
        return [
            self.eta_min + (base_lr - self.eta_min) * (1 + math.cos(math.pi * self.last_epoch / self.T_max)) / 2
            for base_lr in self.base_lrs
        ]
