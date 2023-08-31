import os
import torch
import matplotlib.pyplot as plt
import numpy as np
from torch import optim, nn
from torch.utils.data import Dataset, random_split, DataLoader, ConcatDataset
from sklearn.model_selection import KFold
from .train_utils_tes import *
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score
import seaborn as sns
from scipy.special import softmax
import warnings
import pytorch_lightning as pl
from torchmetrics import AUROC, F1Score, Accuracy

k_folds = 5
kf = KFold(n_splits=k_folds, shuffle=True)
logger = pl.loggers.TensorBoardLogger("logs/")


class MySys(pl.LightningModule):
    def __init__(self, model_name, num_class, pretrain):
        super().__init__()
        self.model = initialize_model(model_name, num_class, pretrain=pretrain)
        self.criterion = CrossEntropyLossOneHot()
        # self.criterion = nn.CrossEntropyLoss()
        self.train_accuracy = Accuracy()
        self.val_accuracy = Accuracy()
        self.train_loss = torch.tensor(0.0)  # Placeholder for training loss
        self.val_loss = torch.tensor(0.0)
        self.auc = AUROC(num_classes=num_class)  # Assuming binary classification, modify for multi-class
        self.f1 = F1Score(num_classes=num_class)

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
        self.scheduler = WarmRestart(self.optimizer, T_max=10, T_mult=1, eta_min=1e-5)
        return [self.optimizer], [self.scheduler]

    def training_step(self, batch, batch_idx):
        images, labels, index = batch
        # 损失Onehot
        output = self(images)
        loss = self.criterion(output, labels)

        # 一般损失
        # output = self(images)
        # loss = self.criterion(output, labels)
        self.train_loss = loss
        preds = torch.argmax(output, dim=1)
        self.train_accuracy(preds, labels)
        return loss

    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        self.log('train_loss', avg_loss, prog_bar=True)
        self.log('train_accuracy', self.train_accuracy.compute(), prog_bar=True)

    def validation_step(self, batch, batch_idx):
        images, labels, index = batch
        # 损失Onehot
        output = self(images)
        loss = self.criterion(output, labels)

        # 一般损失
        # output = self(images)
        # loss = self.criterion(output, labels)

        self.val_loss = loss
        preds = torch.argmax(output, dim=1)
        self.val_accuracy(preds, labels)
        self.auc(output, labels)
        self.f1(output, labels)
        return loss

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        self.log('val_loss', avg_loss, prog_bar=True)
        self.log('val_accuracy', self.val_accuracy.compute(), prog_bar=True)
        self.log('val_auc', self.auc.compute(), prog_bar=True)
        self.log('val_f1', self.f1.compute(), prog_bar=True)


class System():
    def __init__(self, batch_size, model_name, pretrain, num_class, dataset_name, frac, cuda, num_epochs, EOPCH,
                 num_workers):
        # 初始化参数
        self.data_name = dataset_name
        self.batch_size = batch_size
        self.pretrain = pretrain
        self.num_class = num_class
        self.EPOCH = EOPCH
        self.num_workers = num_workers
        self.device = self.device = torch.device("cuda:{}".format(cuda) if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        self.num_epochs = num_epochs
        # self.model = initialize_model(model_name, num_class, pretrain=pretrain)
        self.model = MySys(model_name, num_class, pretrain)
        self.dataset = load_data(dataset_name, frac=frac)
        self.labeled_dataset, self.unlabeled_dataset = random_split(self.dataset, [0.85, 0.15])
        self.transforms = generate_transform(dataset_name)
        # 打印初始化的参数
        # print(
        #     "Model Name: {}\nPretrained: {}\nDataset Name: {}\nDataset Num: {}\nCUDA: {}\nBatch Size: {}\nLabel/UnLabel:{}/{}\n".format(
        #         model_name,
        #         pretrain,
        #         dataset_name, len(self.dataset),
        #         cuda, batch_size, len(self.labeled_dataset), len(self.unlabeled_dataset)))
        # self.optimizer = optim.Adam(self.model.parameters(), lr=0.0001, betas=(0.9, 0.999), eps=1e-08,
        #                             weight_decay=0)
        # self.scheduler = WarmRestart(self.optimizer, T_max=10, T_mult=1, eta_min=1e-5)
        # self.model = self.model.to(self.device)
        # self.criterion = nn.CrossEntropyLoss()

        # self.train_loss_plt, self.train_acc_plt, self.val_loss_plt, self.val_acc_plt, self.f1_score, self.auc = np.array(
        #     []), np.array([]), np.array([]), np.array([]), np.array([]), np.array([])
        #
        # self.avg_train_loss_plt, self.avg_train_acc_plt, self.avg_val_loss_plt, self.avg_val_acc_plt, self.avg_f1_score, self.avg_auc = [
        #     np.zeros(self.num_epochs) for _ in range(6)]
        #
        # self.pse_avg_train_loss_plt, self.pse_avg_train_acc_plt, self.pse_avg_val_loss_plt, self.pse_avg_val_acc_plt, self.pse_avg_f1_score, self.pse_avg_auc = [
        #     np.zeros(self.num_epochs) for _ in range(6)]

        self.trainer = pl.Trainer(
            gpus=1,
            max_epochs=100,
            early_stop_callback=None,
            checkpoint_callback=None,
            logger=logger  # 设置TensorBoardLogger为日志记录器
        )
        # torch.save(self.model.state_dict(), f'models/{self.pretrain}_{self.model_name}ini.pth')

    def plt_avg_label_everything(self, flag):
        if flag:
            self.avg_train_loss_plt += self.train_loss_plt
            self.avg_train_acc_plt += self.train_acc_plt
            self.avg_val_loss_plt += self.val_loss_plt
            self.avg_val_acc_plt += self.val_acc_plt
            self.avg_f1_score += self.f1_score
            self.avg_auc += self.auc
        else:  # 五折训练完成
            self.avg_train_loss_plt, self.avg_train_acc_plt, self.avg_val_loss_plt, self.avg_val_acc_plt, self.avg_f1_score, self.avg_auc = self.avg_train_loss_plt / 5, self.avg_train_acc_plt / 5, self.avg_val_loss_plt / 5, self.avg_val_acc_plt / 5, self.avg_f1_score / 5, self.avg_auc / 5

            print('avg----------------', self.avg_train_loss_plt, self.avg_train_acc_plt, self.avg_val_loss_plt,
                  self.avg_val_acc_plt, self.avg_f1_score, self.avg_auc)

            if self.pretrain:
                save_path = f'results/{self.data_name}/{self.model_name}/pretrain/{self.EPOCH}/'
            else:
                save_path = f'results/{self.data_name}/{self.model_name}/no_pretrain/{self.EPOCH}/'
            # --------------------------------loss
            fig = plt.figure()
            plt.plot(self.avg_train_loss_plt, label='train_loss')
            plt.plot(self.avg_val_loss_plt, label='val_loss')
            plt.ylabel('Loss')
            plt.ylim(0, 0.3)
            plt.xlabel('Epochs')
            plt.xlim(0, self.num_epochs)
            plt.legend()
            plt.savefig(save_path + 'loos.jpg')
            plt.clf()
            plt.close(fig)
            # --------------------------------acc
            fig = plt.figure()
            plt.plot(self.avg_train_acc_plt, label='train_acc')
            plt.plot(self.avg_val_acc_plt, label='val_acc')
            plt.ylabel('Acc')
            plt.ylim(80, 100)
            plt.xlabel('Epochs')
            plt.xlim(0, self.num_epochs)
            plt.legend()
            plt.savefig(save_path + 'acc.jpg')
            # 绘制完毕，清空
            plt.clf()
            plt.close(fig)
            # --------------------------------f1
            fig = plt.figure()
            plt.plot(self.avg_f1_score)
            plt.ylabel('F1')
            plt.ylim(0.8, 1)
            plt.xlabel('Epochs')
            plt.savefig(save_path + 'f1_score.jpg')
            # 绘制完毕，清空
            plt.clf()
            plt.close(fig)
            # --------------------------------auc
            fig = plt.figure()
            plt.plot(self.avg_auc)
            plt.ylabel('Auc')
            plt.ylim(0.9, 1)
            plt.xlabel('Epochs')
            plt.savefig(save_path + 'auc.jpg')
            # 绘制完毕，清空
            plt.clf()
            plt.close(fig)
        self.train_acc_plt, self.train_loss_plt, self.val_acc_plt, self.val_loss_plt, self.f1_score, self.auc = np.array(
            []), np.array([]), np.array([]), np.array([]), np.array([]), np.array([])

    def plt_avg_unlabel_everything(self, flag):
        if flag:
            self.pse_avg_train_loss_plt += self.train_loss_plt
            self.pse_avg_train_acc_plt += self.train_acc_plt
            self.pse_avg_val_loss_plt += self.val_loss_plt
            self.pse_avg_val_acc_plt += self.val_acc_plt
            self.pse_avg_f1_score += self.f1_score
            self.pse_avg_auc += self.auc
        else:
            self.pse_avg_train_loss_plt /= 5
            self.pse_avg_train_acc_plt /= 5
            self.pse_avg_val_loss_plt /= 5
            self.pse_avg_val_acc_plt /= 5
            self.pse_avg_f1_score /= 5
            self.pse_avg_auc /= 5
            print('pse_avg----------------', self.pse_avg_train_loss_plt, self.pse_avg_train_acc_plt,
                  self.pse_avg_val_loss_plt,
                  self.pse_avg_val_acc_plt, self.pse_avg_f1_score, self.pse_avg_auc)
            if self.pretrain:
                save_path = f'results/{self.data_name}/{self.model_name}/pretrain/{self.EPOCH}/pse_'
            else:
                save_path = f'results/{self.data_name}/{self.model_name}/no_pretrain/{self.EPOCH}/pse_'
                # --------------------------------loss
            fig = plt.figure()
            plt.plot(self.pse_avg_train_loss_plt, label='train_loss')
            plt.plot(self.pse_avg_val_loss_plt, label='val_loss')
            plt.ylabel('Loss')
            plt.ylim(0, 0.3)
            plt.xlabel('Epochs')
            plt.xlim(0, self.num_epochs)
            plt.legend()
            plt.savefig(save_path + 'loos.jpg')
            plt.clf()
            plt.close(fig)
            # --------------------------------acc
            fig = plt.figure()
            plt.plot(self.pse_avg_train_acc_plt, label='train_acc')
            plt.plot(self.pse_avg_val_acc_plt, label='val_acc')
            plt.ylabel('Acc')
            plt.ylim(80, 100)
            plt.xlabel('Epochs')
            plt.xlim(0, self.num_epochs)
            plt.legend()
            plt.savefig(save_path + 'acc.jpg')
            # 绘制完毕，清空
            plt.clf()
            plt.close(fig)
            # --------------------------------f1
            fig = plt.figure()
            plt.plot(self.pse_avg_f1_score)
            plt.ylabel('F1')
            plt.ylim(0.8, 1)
            plt.xlabel('Epochs')
            plt.savefig(save_path + 'f1_score.jpg')
            # 绘制完毕，清空
            plt.clf()
            plt.close(fig)
            # --------------------------------auc
            fig = plt.figure()
            plt.plot(self.pse_avg_auc)
            plt.ylabel('Auc')
            plt.ylim(0.9, 1)
            plt.xlabel('Epochs')
            plt.savefig(save_path + 'auc.jpg')
            # 绘制完毕，清空
            plt.clf()
            plt.close(fig)
        self.train_acc_plt, self.train_loss_plt, self.val_acc_plt, self.val_loss_plt, self.f1_score, self.auc = np.array(
            []), np.array([]), np.array([]), np.array([]), np.array([]), np.array([])

    def save_model(self, name):
        dir_path = f'models/{self.data_name}/{self.model_name}'
        os.makedirs(os.path.dirname(dir_path), exist_ok=True)
        torch.save(self.model.state_dict(), dir_path + name)

    def load_model(self, name):
        dir_path = f'models/{self.data_name}/{self.model_name}' + name
        self.model.load_state_dict(torch.load(dir_path))

    # 绘制损失、准确率曲线
    def plt_loss_acc(self, fold_i):
        if self.pretrain:
            save_path = f'results/{self.data_name}/{self.model_name}/pretrain/{self.EPOCH}/{fold_i}/'
        else:
            save_path = f'results/{self.data_name}/{self.model_name}/no_pretrain/{self.EPOCH}/{fold_i}/'

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        # --------------------------------loss
        fig = plt.figure()
        plt.plot(self.train_loss_plt, label='train_loss')
        plt.plot(self.val_loss_plt, label='val_loss')
        plt.ylabel('Loss')
        plt.ylim(0, 1)
        plt.xlabel('Epochs')
        plt.xlim(0, self.num_epochs)
        plt.legend()
        plt.savefig(save_path + 'loos.jpg')
        plt.clf()
        plt.close(fig)
        # --------------------------------acc
        fig = plt.figure()
        plt.plot(self.train_acc_plt, label='train_acc')
        plt.plot(self.val_acc_plt, label='val_acc')
        plt.ylabel('Acc')
        plt.ylim(80, 100)
        plt.xlabel('Epochs')
        plt.xlim(0, self.num_epochs)
        plt.legend()
        plt.savefig(save_path + 'acc.jpg')
        # 绘制完毕，清空
        plt.clf()
        plt.close(fig)
        # --------------------------------f1
        fig = plt.figure()
        plt.plot(self.f1_score)
        plt.ylabel('F1')
        plt.ylim(0.8, 1)
        plt.xlabel('Epochs')
        plt.savefig(save_path + 'f1_score.jpg')
        # 绘制完毕，清空
        plt.clf()
        plt.close(fig)
        # --------------------------------auc
        fig = plt.figure()
        plt.plot(self.auc)
        plt.ylabel('Auc')
        plt.ylim(0.9, 1)
        plt.xlabel('Epochs')
        plt.savefig(save_path + 'auc.jpg')
        # 绘制完毕，清空
        plt.clf()
        plt.close(fig)

        print('\ntran_acc', self.train_acc_plt, '\ntran_loss', self.train_loss_plt, '\nval_acc', self.val_acc_plt,
              '\nval_loss',
              self.val_loss_plt)

    def eval(self, images, label):
        if self.model_name == 'inception_v3':
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
            # print(output.shape)
            loss = self.criterion(output, label)

        # _, pred = torch.max(output, 1)  # 前一个是输出的最大概率值是多少，后一个是最大概率对应的标签即类别
        return loss, output

    def plt_conf_matrix(self, conf_matrix, pse):
        # 如果画不出来图，就是缺少了类别
        warnings.filterwarnings('ignore')
        if self.pretrain:
            save_path = f'results/{self.data_name}/{self.model_name}/pretrain/{self.EPOCH}/matrix/'
        else:
            save_path = f'results/{self.data_name}/{self.model_name}/no_pretrain/{self.EPOCH}/matrix/'
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        if self.data_name == 'crop':
            # "返青" - Regreening
            # "分蘖" - Tillering
            # "抽穗" - Heading
            # "拔节" - Booting
            # "乳熟" - Milk ripening
            l = ["Regreening", "Tillering", "Heading", "Booting", 'Milk ripening']
        elif self.data_name == 'apple':
            # dit = {0: "healthy", 1: "multiple", 2: "rust", 3: "scab"}
            l = ["healthy", "multiple", "rust", "scab"]
        else:
            # dit = {0: "Black-grass", 1: "Charlock", 2: "Cleavers", 3: "Chickweed", 4: "wheat", 5: "Fat Hen",
            #        6: "Loose Silky-bent", 7: "Maize", 8: "nonsegmentedv2", 9: "Scentless Mayweed", 10: "Shepherd Purse",
            #        11: "Small-flowered Cranesbill", 12: "Sugar beet"}
            l = ["Black-grass", "Charlock", "Cleavers", "Chickweed", "wheat", "Fat Hen", "Loose Silky-bent", "Maize",
                 "Scentless Mayweed", "Shepherd Purse", "Small-flowered Cranesbill", "Sugar beet"]

        fig = plt.figure()
        sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", cbar=False)
        plt.xticks(np.arange(len(l)), l, rotation=45)
        plt.yticks(np.arange(len(l)), l, rotation=0)
        plt.xlabel('Predicted label')
        plt.ylabel('True label')
        plt.title('Confusion Matrix')
        plt.tight_layout(rect=[0, 0, 0.85, 1])
        plt.savefig(save_path + pse + 'conf_matrix.png')
        plt.clf()
        plt.close(fig)

    # 不要计算梯度，否则测试的时候内存会爆掉，我们确定不会调用Tensor.backward()函数
    def labeling(self, pseudo_dataset, name):
        self.load_model(name)
        pseudo_dataset = TransformedDataset(pseudo_dataset, self.transforms['val_transforms'])
        pseudo_dataloader = DataLoader(pseudo_dataset, batch_size=self.batch_size, shuffle=True,
                                       num_workers=self.num_workers)
        self.model.eval()
        with torch.no_grad():
            for images, labels, index in pseudo_dataloader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                _, predicted = torch.max(outputs.data, 1)
                for i in range(len(index)):
                    self.dataset.labels[index[i]] = predicted[i].item()

    def train(self, train_dataset, val_dataset, save_name, pse, fold_i):
        train_dataset = TransformedDataset(train_dataset, self.transforms['train_transforms'])
        val_dataset = TransformedDataset(val_dataset, self.transforms['val_transforms'])

        # 创建数据加载器
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers,
                                  drop_last=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True)

        best_acc, total_true_labels, total_predicted_labels = 0, [], []
        self.trainer.fit(self.model, train_dataloader, val_dataloader)

        # for epoch in range(self.num_epochs):
        #     train_loss, train_correct, train_total = 0, 0, 0
        #     # 设置模型训练状态
        #     self.model.train()
        #     for images, labels, _ in train_loader:
        #         images, labels = images.to(self.device), labels.to(self.device)
        #         self.optimizer.zero_grad()  # 梯度清零
        #         loss, output = self.eval(images, labels)  # 预测结果
        #         _, predicted = torch.max(output, 1)
        #         loss.backward()  # 计算梯度值
        #         self.optimizer.step()  # 更新权重
        #         train_total += labels.size(0)
        #         train_correct += (predicted == labels).sum().item()
        #         train_loss += loss.item()
        #
        #     self.model.eval()
        #     val_loss, val_correct, val_total, true_labels, predicted_labels, probabilities = 0.0, 0, 0, [], [], []
        #     with torch.no_grad():
        #         for images, labels, _ in val_loader:
        #             images, labels = images.to(self.device), labels.to(self.device)
        #             loss, output = self.eval(images, labels)
        #             _, predicted = torch.max(output, 1)
        #             probabilities.extend(subset.tolist() for subset in torch.softmax(output, dim=1).cpu())
        #
        #             # 统计验证集相关
        #             val_loss += loss.item()
        #             val_total += labels.size(0)
        #             val_correct += (predicted == labels).sum().item()
        #             # 统计标签数值
        #             true_labels.extend(labels.cpu().numpy())
        #             total_true_labels.extend(labels.cpu().numpy())
        #             predicted_labels.extend(predicted.cpu().numpy())
        #             total_predicted_labels.extend(predicted.cpu().numpy())
        #     self.scheduler.step()
        #
        #     # 统计acc,loss
        #     train_loss /= len(train_loader)
        #     train_accuracy = 100.0 * train_correct / train_total
        #     val_loss /= len(val_loader)
        #     val_accuracy = 100.0 * val_correct / val_total
        #     # 统计f1,auc
        #     f1 = f1_score(true_labels, predicted_labels, average='macro')
        #     try:
        #         auc = roc_auc_score(true_labels, probabilities, multi_class='ovo')
        #     except ValueError:
        #         print('验证集缺少某些类别，已处理！')
        #         missing_classes = list(set(range(self.num_class)) - set(true_labels))
        #         probabilities = np.delete(probabilities, missing_classes, axis=1)
        #         probabilities = softmax(probabilities, axis=1)
        #         auc = roc_auc_score(true_labels, probabilities, multi_class='ovo')
        #     # 得到epoch里面
        #     self.f1_score = np.append(self.f1_score, f1)
        #     self.auc = np.append(self.auc, auc)
        #
        #     # print(
        #     #     f"Epoch {epoch + 1}/{self.num_epochs}: Train Loss: {train_loss:.4f} | Train Acc: {train_accuracy:.2f}% | Val Loss: {val_loss:.4f} | Val Acc: {val_accuracy:.2f}%")
        #     # if train_accuracy > best_acc:
        #     #     best_acc = train_accuracy
        #     #     self.save_model(save_name)
        #     if val_accuracy > best_acc:
        #         best_acc = val_accuracy
        #         self.save_model(save_name)
        #
        #     print(
        #         f"Epoch {epoch + 1}/{self.num_epochs}: Train Loss: {train_loss:.4f} | Train Acc: {train_accuracy:.2f}% | Val Loss: {val_loss:.4f} | Val Acc: {val_accuracy:.2f}% | --Best Acc: {best_acc:.2f}%")
        #     # append相关参数以便绘制结果图
        #     self.train_loss_plt, self.train_acc_plt, self.val_loss_plt, self.val_acc_plt = [
        #         np.append(arr, val) for arr, val in
        #         zip([self.train_loss_plt, self.train_acc_plt, self.val_loss_plt, self.val_acc_plt, self.auc],
        #             [train_loss, train_accuracy, val_loss, val_accuracy])]

        # 统计f1,auc,conf_matrix

        # conf_matrix = confusion_matrix(total_true_labels, total_predicted_labels)
        # if pse:
        #     self.plt_conf_matrix(conf_matrix, f'_{fold_i}_pse_')
        # else:
        #     self.plt_conf_matrix(conf_matrix, f'_{fold_i}_')

    def run(self):
        for fold, (train_indices, val_indices) in enumerate(kf.split(self.labeled_dataset)):
            print(f"Fold {fold + 1}/{k_folds}")
            self.model.load_state_dict(
                torch.load(f'models/{self.pretrain}_{self.model_name}ini.pth', map_location=self.device))
            pre = 'pre' if self.pretrain else 'nopre'
            save_name = f'{pre}_{fold + 1}fold.pth'
            train_dataset = torch.utils.data.Subset(self.labeled_dataset, train_indices)
            val_dataset = torch.utils.data.Subset(self.labeled_dataset, val_indices)
            print(f'train_data/val_data is {len(train_dataset)}/{len(val_dataset)}')
            self.train(train_dataset, val_dataset, save_name, False, f'fold{fold + 1}')
            self.plt_loss_acc(f'fold{fold + 1}')
            self.plt_avg_label_everything(flag=True)

            # 打伪标签
            self.labeling(self.unlabeled_dataset, save_name)

            # 混合数据集
            mix_dataset = ConcatDataset([train_dataset, self.unlabeled_dataset])

            # 利用含有伪标签的混合数据进行训练
            save_name = f'{pre}_{fold + 1}fold_pseudo.pth'
            print('----------------------------\t进行伪标签训练')
            self.train(mix_dataset, val_dataset, save_name, True, f'fold{fold + 1}')

            self.plt_loss_acc(f'fold{fold + 1}_pse')
            self.plt_avg_unlabel_everything(flag=True)

        self.plt_avg_label_everything(flag=False)
        self.plt_avg_unlabel_everything(flag=False)
