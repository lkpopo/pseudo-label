import os
import torch
import matplotlib.pyplot as plt
import numpy as np
from torch import optim, nn
from torch.utils.data import Dataset, random_split, DataLoader, ConcatDataset
from sklearn.model_selection import KFold
from bak.train_utils_tes import *
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score
import seaborn as sns
from scipy.special import softmax
import warnings
import pytorch_lightning as pl
from torchmetrics import AUROC, F1Score, Accuracy
import random
from utils.lrs_scheduler import *
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
import gc

k_folds = 5
folds = KFold(n_splits=k_folds, shuffle=True, random_state=2023)
logger = pl.loggers.TensorBoardLogger("logs/")


def seed_reproducer(seed=2023):
    """Reproducer for pytorch experiment.

    Parameters
    ----------
    seed: int, optional (default = 2019)
        Radnom seed.

    Example
    -------
    seed_reproducer(seed=2019).
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.enabled = True


class MySys(pl.LightningModule):
    def __init__(self, model_name, num_class, pretrain):
        super().__init__()
        self.model = initialize_model(model_name, num_class, pretrain=pretrain)
        self.criterion = CrossEntropyLossOneHot()
        self.train_accuracy = Accuracy(task="multiclass", num_classes=num_class)
        self.val_accuracy = Accuracy(task="multiclass", num_classes=num_class)
        self.auc = AUROC(task="multiclass",
                         num_classes=num_class)  # Assuming binary classification, modify for multi-class
        self.f1 = F1Score(task="multiclass", num_classes=num_class)

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
        self.scheduler = WarmRestart(self.optimizer, T_max=10, T_mult=1, eta_min=1e-5)
        return [self.optimizer], [self.scheduler]

    def training_step(self, batch, batch_idx):
        images, labels, index = batch
        output = self(images)
        loss = self.criterion(output, labels)
        self.train_accuracy(torch.argmax(output, dim=1), labels)
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def on_train_epoch_end(self):
        self.log('train_accuracy', self.train_accuracy.compute(), prog_bar=True)
        train_losses = self.trainer.logged_metrics.get('train_loss', [])
        avg_train_loss = torch.tensor(train_losses).mean()
        self.log('avg_train_loss', avg_train_loss, prog_bar=True)
        self.current_epoch += 1
        if self.current_epoch < (self.trainer.max_epochs - 4):
            self.scheduler = warm_restart(self.scheduler, T_mult=2)

    def validation_step(self, batch, batch_idx):
        images, labels, index = batch
        output = self(images)
        loss = self.criterion(output, labels)
        self.log('val_loss', loss, prog_bar=True)
        self.val_accuracy(torch.argmax(output, dim=1), labels)
        self.auc(output, labels)
        self.f1(output, labels)

        return loss

    def on_validation_epoch_end(self):
        val_losses = self.trainer.logged_metrics.get('val_loss', [])
        avg_val_loss = torch.tensor(val_losses).mean()
        self.log('avg_val_loss', avg_val_loss, prog_bar=True)
        self.log('val_accuracy', self.val_accuracy.compute(), prog_bar=True)
        self.log('val_auc', self.auc.compute(), prog_bar=True)
        self.log('val_f1', self.f1.compute(), prog_bar=True)


batch_size = 4
num_workers = 4

if __name__ == "__main__":
    seed_reproducer(2023)
    data = load_data("apple", frac=1)
    transforms = generate_transform("apple")
    for fold_i, (train_index, val_index) in enumerate(folds.split(data)):
        train_data = torch.utils.data.Subset(data, train_index)
        val_data = torch.utils.data.Subset(data, val_index)
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                                  drop_last=True)
        val_loader = DataLoader(val_data, batch_size=batch_size, num_workers=num_workers, pin_memory=True)

        model = MySys("se_resnet50", 4, True)

        trainer = pl.Trainer(
            max_epochs=70,
            accelerator="gpu",
            # early_stop_callback=None,
            # checkpoint_callback=None,
            logger=logger  # 设置TensorBoardLogger为日志记录器
        )
        trainer.fit(model, train_loader, val_loader)
        del model
        gc.collect()
        torch.cuda.empty_cache()
