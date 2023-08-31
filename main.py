from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QAction, QVBoxLayout, QWidget
from PyQt5.QtCore import Qt  # 导入Qt模块
from PyQt5.QtGui import QIcon
from ui import Ui_MainWindow  # 导入生成的界面类
import sys

sys.path.append('utils')
from argparse import ArgumentParser
from utils.train import System
import threading
import ctypes, inspect


def _async_raise(tid, exctype):
    """raises the exception, performs cleanup if needed"""
    tid = ctypes.c_long(tid)
    if not inspect.isclass(exctype):
        exctype = type(exctype)
    res = ctypes.pythonapi.PyThreadState_SetAsyncExc(tid, ctypes.py_object(exctype))
    if res == 0:
        raise ValueError("invalid thread id")
    elif res != 1:
        # """if it returns a number greater than one, you're in trouble,
        # and you should call it again with exc=NULL to revert the effect"""
        ctypes.pythonapi.PyThreadState_SetAsyncExc(tid, None)
        raise SystemError("PyThreadState_SetAsyncExc failed")

def init_hparams(backbone, batch_size, num_workers, epochs, cuda, num_class, frac, pretrain, dataset):
    parser = ArgumentParser(add_help=False)
    parser.add_argument("--backbone", type=str, default=backbone)
    parser.add_argument("--batch_size", type=int, default=batch_size)
    parser.add_argument("--num_workers", type=int, default=num_workers)
    parser.add_argument("--epochs", type=int, default=epochs)
    parser.add_argument("--cuda", type=int, default=cuda)
    parser.add_argument("--num_class", type=int, default=num_class)
    parser.add_argument("--frac", type=float, default=frac)
    parser.add_argument("--pretrain", type=bool, default=pretrain)
    parser.add_argument("--dataset", type=str, default=dataset)
    try:
        hparams = parser.parse_args()
    except:
        print('解析超参数失败，请检查超参数设置')
        hparams = parser.parse_args([])

    return hparams


class HelpWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.init_ui()

    def init_ui(self):
        self.setGeometry(150, 150, 300, 200)
        self.setWindowTitle('Help')
        self.setWindowIcon(QIcon("utils/icon.jpg"))

        help_label = QLabel("Backbone: 选取主干网\n"
                            "Batch-size: 设置训练时每个batch的大小\n"
                            "Num-workers: 用多线程加载数据，一般仅在linux上使用，默认请设置为1\n"
                            "Epochs: 训练的轮数\n"
                            "Frac: 快速训练，在0-1之间，表示取用数据集的多少进行快速训练，用于验证整个训练流程的可行性\n"
                            "Cuda: 选取第几块gpu，默认请设置为0，如果没有gpu会设置为cpu\n"
                            "Pretrain: 是否采取预训练的模型\n"
                            "Dataset: 数据集的在data下的路径\n"
                            , self)

        help_label.setStyleSheet("font-size:20px;")

        help_label.setAlignment(Qt.AlignLeft)

        layout = QVBoxLayout()
        layout.addWidget(help_label)

        central_widget = QWidget()
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)


class OutputRedirector:
    def __init__(self, text_widget):
        self.text_widget = text_widget

    def write(self, text):
        self.text_widget.insertPlainText(text)


# noinspection PyPep8Naming
class MyWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = Ui_MainWindow()  # 创建界面对象
        self.ui.setupUi(self)  # 初始化界面
        self.ui.Start_train_button.clicked.connect(self.start)
        self.ui.Stop_train_button.clicked.connect(self.stop)

        menubar = self.menuBar()
        help_action = QAction('Help', self)
        help_action.triggered.connect(self.show_help_window)
        menubar.addAction(help_action)
        self.help_window = None

        self.thread = None
        self.ui.out_window.setReadOnly(True)
        sys.stdout = OutputRedirector(self.ui.out_window)
        sys.stderr = OutputRedirector(self.ui.out_window)

    def show_help_window(self):
        if not self.help_window:  # 如果窗口尚未创建，创建并显示
            self.help_window = HelpWindow()
            self.help_window.show()
        else:
            self.help_window.show()  # 如果窗口已创建，直接显示

    def run(self):
        Backbone = str(self.ui.Backbone.currentText())
        Batch_size = int(self.ui.Batch_size.text())
        Num_workers = int(self.ui.Num_workers.text())
        Epochs = int(self.ui.Epochs.text())
        Num_class = int(self.ui.Num_class.text())
        Cuda = int(self.ui.Cuda.text())
        Frac = float(self.ui.Frac.text())
        Pretrain = bool(self.ui.Pretrain.text() == "True")
        Dataset = str(self.ui.Dataset.text())
        hparams = init_hparams(Backbone, Batch_size, Num_workers, Epochs, Cuda, Num_class, Frac, Pretrain, Dataset)
        # hparams = init_hparams(backbone="inception_v3", batch_size=6, num_workers=1, epochs=4, cuda=0, num_class=4,
        #                        frac=0.3,
        #                        pretrain=True, dataset="apple")
        for EPOCH in range(1):
            mySys = System(hparams.batch_size, hparams.backbone, hparams.pretrain, hparams.num_class,
                           hparams.dataset,
                           hparams.frac, hparams.cuda,
                           hparams.epochs, EPOCH, hparams.num_workers)
            mySys.run()

    def start(self):
        self.thread = threading.Thread(target=self.run)
        self.thread.start()

    def stop(self):
        _async_raise(self.thread.ident, SystemExit)
        print("Mission Stopped!")


if __name__ == '__main__':
    if sys.platform.startswith('win'):
        app = QApplication(sys.argv)
        window = MyWindow()
        window.show()
        sys.exit(app.exec_())
    else:
        print("please input following parameters:")
        backbone = input("1. Backbone (inception_v3 vgg16 resnet50 ...): ")
        batch_size = int(input("2. Batch size: "))
        num_workers = int(input("3. Number of workers: "))
        epochs = int(input("4. Epochs: "))
        cuda = int(input("5. CUDA (0 or 1): "))
        num_class = int(input("6. Number of classes: "))
        frac = float(input("7. Fraction (True or False): "))
        pretrain = bool(input("8. Pretrained (True or False): "))
        dataset = input("9. Dataset-name: ")

        # 初始化超参数
        hparams = init_hparams(backbone, batch_size, num_workers, epochs, cuda, num_class, frac, pretrain, dataset)
        for EPOCH in range(1):
            mySys = System(hparams.batch_size, hparams.backbone, hparams.pretrain, hparams.num_class, hparams.dataset,
                           hparams.frac, hparams.cuda,
                           hparams.epochs, EPOCH, hparams.num_workers)
            mySys.run()
