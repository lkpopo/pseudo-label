# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file '.\ui.ui'
#
# Created by: PyQt5 UI code generator 5.15.7
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1226, 600)
        MainWindow.setMinimumSize(QtCore.QSize(1226, 600))
        MainWindow.setMaximumSize(QtCore.QSize(1226, 600))
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(".\\utils/icon.jpg"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        MainWindow.setWindowIcon(icon)
        MainWindow.setStyleSheet("QMainWindow {\n"
"    background-color:rgb(43,43,43);\n"
"}")
        MainWindow.setIconSize(QtCore.QSize(50, 50))
        MainWindow.setToolButtonStyle(QtCore.Qt.ToolButtonTextBesideIcon)
        MainWindow.setUnifiedTitleAndToolBarOnMac(False)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.Start_train_button = QtWidgets.QPushButton(self.centralwidget)
        self.Start_train_button.setGeometry(QtCore.QRect(90, 450, 101, 41))
        self.Start_train_button.setStyleSheet("QPushButton {\n"
"    font-size: 20px;\n"
"    color: #333;\n"
"    background-color: #f0f0f0;\n"
"    padding: 5px;\n"
"    border: 1px solid #ccc;\n"
"    border-radius: 3px;\n"
"}\n"
"\n"
"QPushButton:hover {\n"
"       background-color: #ddd;\n"
"}\n"
"")
        self.Start_train_button.setObjectName("Start_train_button")
        self.Stop_train_button = QtWidgets.QPushButton(self.centralwidget)
        self.Stop_train_button.setGeometry(QtCore.QRect(230, 450, 101, 41))
        self.Stop_train_button.setStyleSheet("QPushButton {\n"
"    font-size: 20px;\n"
"    color: #333;\n"
"    background-color: #f0f0f0;\n"
"    padding: 5px;\n"
"    border: 1px solid #ccc;\n"
"    border-radius: 3px;\n"
"}\n"
"\n"
"QPushButton:hover {\n"
"       background-color: #ddd;\n"
"}\n"
"")
        self.Stop_train_button.setObjectName("Stop_train_button")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(1020, 540, 211, 41))
        self.label.setStyleSheet("QLabel {\n"
"    color: gray; font-size: 15px;\n"
"    font-family: SimHei;\n"
"}\n"
"")
        self.label.setObjectName("label")
        self.out_window = QtWidgets.QPlainTextEdit(self.centralwidget)
        self.out_window.setGeometry(QtCore.QRect(520, 60, 571, 391))
        self.out_window.setStyleSheet("QPlainTextEdit{\n"
"background-color: #f0f0f0;\n"
"            border: 1px solid #d1d1d1;\n"
"            font-family: Arial, sans-serif;\n"
"            font-size: 16px;\n"
"\n"
"}")
        self.out_window.setObjectName("out_window")
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(730, 20, 161, 31))
        font = QtGui.QFont()
        font.setPointSize(18)
        self.label_2.setFont(font)
        self.label_2.setStyleSheet("QLabel{\n"
"color:rgb(240,240,240);\n"
"}")
        self.label_2.setObjectName("label_2")
        self.layoutWidget = QtWidgets.QWidget(self.centralwidget)
        self.layoutWidget.setGeometry(QtCore.QRect(71, 71, 292, 362))
        self.layoutWidget.setObjectName("layoutWidget")
        self.gridLayout = QtWidgets.QGridLayout(self.layoutWidget)
        self.gridLayout.setContentsMargins(0, 0, 0, 0)
        self.gridLayout.setObjectName("gridLayout")
        self.Epochs_label = QtWidgets.QLabel(self.layoutWidget)
        self.Epochs_label.setStyleSheet("QLabel {\n"
"    font-size: 20px;\n"
"    color: #333;\n"
"    background-color: #f0f0f0;\n"
"    padding: 5px;\n"
"    border: 1px solid #ccc;\n"
"    border-radius: 3px;\n"
"}\n"
"\n"
"QLabel:hover {\n"
"    background-color: #ddd;\n"
"}\n"
"")
        self.Epochs_label.setObjectName("Epochs_label")
        self.gridLayout.addWidget(self.Epochs_label, 3, 0, 1, 1)
        self.Frac_label = QtWidgets.QLabel(self.layoutWidget)
        self.Frac_label.setStyleSheet("QLabel {\n"
"    font-size: 20px;\n"
"    color: #333;\n"
"    background-color: #f0f0f0;\n"
"    padding: 5px;\n"
"    border: 1px solid #ccc;\n"
"    border-radius: 3px;\n"
"}\n"
"\n"
"QLabel:hover {\n"
"    background-color: #ddd;\n"
"}\n"
"")
        self.Frac_label.setObjectName("Frac_label")
        self.gridLayout.addWidget(self.Frac_label, 5, 0, 1, 1)
        self.Frac = QtWidgets.QLineEdit(self.layoutWidget)
        self.Frac.setStyleSheet("/* Style for QLineEdit */\n"
"QLineEdit {\n"
"    font-size: 20px;\n"
"    padding: 5px;\n"
"    border: 1px solid #ccc;\n"
"    border-radius: 3px;\n"
"}\n"
"\n"
"QLineEdit:focus {\n"
"    border-color: #007BFF;\n"
"}")
        self.Frac.setObjectName("Frac")
        self.gridLayout.addWidget(self.Frac, 5, 1, 1, 1)
        self.Num_workers_label = QtWidgets.QLabel(self.layoutWidget)
        self.Num_workers_label.setStyleSheet("QLabel {\n"
"    font-size: 20px;\n"
"    color: #333;\n"
"    background-color: #f0f0f0;\n"
"    padding: 5px;\n"
"    border: 1px solid #ccc;\n"
"    border-radius: 3px;\n"
"}\n"
"\n"
"QLabel:hover {\n"
"    background-color: #ddd;\n"
"}\n"
"")
        self.Num_workers_label.setObjectName("Num_workers_label")
        self.gridLayout.addWidget(self.Num_workers_label, 2, 0, 1, 1)
        self.Num_class = QtWidgets.QLineEdit(self.layoutWidget)
        self.Num_class.setStyleSheet("/* Style for QLineEdit */\n"
"QLineEdit {\n"
"    font-size: 20px;\n"
"    padding: 5px;\n"
"    border: 1px solid #ccc;\n"
"    border-radius: 3px;\n"
"}\n"
"\n"
"QLineEdit:focus {\n"
"    border-color: #007BFF;\n"
"}")
        self.Num_class.setObjectName("Num_class")
        self.gridLayout.addWidget(self.Num_class, 4, 1, 1, 1)
        self.Pretrain_label = QtWidgets.QLabel(self.layoutWidget)
        self.Pretrain_label.setStyleSheet("QLabel {\n"
"    font-size: 20px;\n"
"    color: #333;\n"
"    background-color: #f0f0f0;\n"
"    padding: 5px;\n"
"    border: 1px solid #ccc;\n"
"    border-radius: 3px;\n"
"}\n"
"\n"
"QLabel:hover {\n"
"    background-color: #ddd;\n"
"}\n"
"")
        self.Pretrain_label.setObjectName("Pretrain_label")
        self.gridLayout.addWidget(self.Pretrain_label, 7, 0, 1, 1)
        self.Dataset = QtWidgets.QLineEdit(self.layoutWidget)
        self.Dataset.setToolTip("")
        self.Dataset.setStyleSheet("/* Style for QLineEdit */\n"
"QLineEdit {\n"
"    font-size: 20px;\n"
"    padding: 5px;\n"
"    border: 1px solid #ccc;\n"
"    border-radius: 3px;\n"
"}\n"
"\n"
"QLineEdit:focus {\n"
"    border-color: #007BFF;\n"
"}")
        self.Dataset.setObjectName("Dataset")
        self.gridLayout.addWidget(self.Dataset, 8, 1, 1, 1)
        self.Backbone = QtWidgets.QComboBox(self.layoutWidget)
        self.Backbone.setStyleSheet("/* Style for QComboBox */\n"
"QComboBox {\n"
"    font-size: 18px;\n"
"    padding: 5px;\n"
"    border: 1px solid #ccc;\n"
"    border-radius: 3px;\n"
"    background-color: white;\n"
"    selection-background-color: #007BFF;\n"
"    selection-color: white;\n"
"}\n"
"\n"
"QComboBox::drop-down {\n"
"    subcontrol-origin: padding;\n"
"    subcontrol-position: top right;\n"
"    width: 20px;\n"
"    border-left-width: 1px;\n"
"    border-left-color: #ccc;\n"
"    border-left-style: solid;\n"
"    border-top-right-radius: 3px;\n"
"    border-bottom-right-radius: 3px;\n"
"    background-color: #f0f0f0;\n"
"}\n"
"\n"
"QComboBox QAbstractItemView {\n"
"    border: 1px solid #ccc;\n"
"    selection-background-color: #007BFF;\n"
"    selection-color: white;\n"
"}")
        self.Backbone.setObjectName("Backbone")
        self.Backbone.addItem("")
        self.Backbone.addItem("")
        self.Backbone.addItem("")
        self.Backbone.addItem("")
        self.Backbone.addItem("")
        self.Backbone.addItem("")
        self.Backbone.addItem("")
        self.Backbone.addItem("")
        self.gridLayout.addWidget(self.Backbone, 0, 1, 1, 1)
        self.Num_workers = QtWidgets.QLineEdit(self.layoutWidget)
        self.Num_workers.setStyleSheet("/* Style for QLineEdit */\n"
"QLineEdit {\n"
"    font-size: 20px;\n"
"    padding: 5px;\n"
"    border: 1px solid #ccc;\n"
"    border-radius: 3px;\n"
"}\n"
"\n"
"QLineEdit:focus {\n"
"    border-color: #007BFF;\n"
"}")
        self.Num_workers.setObjectName("Num_workers")
        self.gridLayout.addWidget(self.Num_workers, 2, 1, 1, 1)
        self.Epochs = QtWidgets.QLineEdit(self.layoutWidget)
        self.Epochs.setStyleSheet("/* Style for QLineEdit */\n"
"QLineEdit {\n"
"    font-size: 20px;\n"
"    padding: 5px;\n"
"    border: 1px solid #ccc;\n"
"    border-radius: 3px;\n"
"}\n"
"\n"
"QLineEdit:focus {\n"
"    border-color: #007BFF;\n"
"}")
        self.Epochs.setObjectName("Epochs")
        self.gridLayout.addWidget(self.Epochs, 3, 1, 1, 1)
        self.Pretrain = QtWidgets.QLineEdit(self.layoutWidget)
        self.Pretrain.setStyleSheet("/* Style for QLineEdit */\n"
"QLineEdit {\n"
"    font-size: 20px;\n"
"    padding: 5px;\n"
"    border: 1px solid #ccc;\n"
"    border-radius: 3px;\n"
"}\n"
"\n"
"QLineEdit:focus {\n"
"    border-color: #007BFF;\n"
"}")
        self.Pretrain.setObjectName("Pretrain")
        self.gridLayout.addWidget(self.Pretrain, 7, 1, 1, 1)
        self.Dataset_label = QtWidgets.QLabel(self.layoutWidget)
        self.Dataset_label.setStyleSheet("QLabel {\n"
"    font-size: 20px;\n"
"    color: #333;\n"
"    background-color: #f0f0f0;\n"
"    padding: 5px;\n"
"    border: 1px solid #ccc;\n"
"    border-radius: 3px;\n"
"}\n"
"\n"
"QLabel:hover {\n"
"    background-color: #ddd;\n"
"}\n"
"")
        self.Dataset_label.setObjectName("Dataset_label")
        self.gridLayout.addWidget(self.Dataset_label, 8, 0, 1, 1)
        self.Batch_size = QtWidgets.QLineEdit(self.layoutWidget)
        self.Batch_size.setStyleSheet("/* Style for QLineEdit */\n"
"QLineEdit {\n"
"    font-size: 20px;\n"
"    padding: 5px;\n"
"    border: 1px solid #ccc;\n"
"    border-radius: 3px;\n"
"}\n"
"\n"
"QLineEdit:focus {\n"
"    border-color: #007BFF;\n"
"}")
        self.Batch_size.setObjectName("Batch_size")
        self.gridLayout.addWidget(self.Batch_size, 1, 1, 1, 1)
        self.Batch_size_label = QtWidgets.QLabel(self.layoutWidget)
        self.Batch_size_label.setStyleSheet("QLabel {\n"
"    font-size: 20px;\n"
"    color: #333;\n"
"    background-color: #f0f0f0;\n"
"    padding: 5px;\n"
"    border: 1px solid #ccc;\n"
"    border-radius: 3px;\n"
"}\n"
"\n"
"QLabel:hover {\n"
"    background-color: #ddd;\n"
"}\n"
"")
        self.Batch_size_label.setObjectName("Batch_size_label")
        self.gridLayout.addWidget(self.Batch_size_label, 1, 0, 1, 1)
        self.Backbone_label = QtWidgets.QLabel(self.layoutWidget)
        self.Backbone_label.setStyleSheet("QLabel {\n"
"    font-size: 20px;\n"
"    color: #333;\n"
"    background-color: #f0f0f0;\n"
"    padding: 5px;\n"
"    border: 1px solid #ccc;\n"
"    border-radius: 3px;\n"
"}\n"
"\n"
"QLabel:hover {\n"
"    background-color: #ddd;\n"
"}\n"
"")
        self.Backbone_label.setObjectName("Backbone_label")
        self.gridLayout.addWidget(self.Backbone_label, 0, 0, 1, 1)
        self.Num_class_label = QtWidgets.QLabel(self.layoutWidget)
        self.Num_class_label.setStyleSheet("QLabel {\n"
"    font-size: 20px;\n"
"    color: #333;\n"
"    background-color: #f0f0f0;\n"
"    padding: 5px;\n"
"    border: 1px solid #ccc;\n"
"    border-radius: 3px;\n"
"}\n"
"\n"
"QLabel:hover {\n"
"    background-color: #ddd;\n"
"}\n"
"")
        self.Num_class_label.setObjectName("Num_class_label")
        self.gridLayout.addWidget(self.Num_class_label, 4, 0, 1, 1)
        self.Cuda_label = QtWidgets.QLabel(self.layoutWidget)
        self.Cuda_label.setStyleSheet("QLabel {\n"
"    font-size: 20px;\n"
"    color: #333;\n"
"    background-color: #f0f0f0;\n"
"    padding: 5px;\n"
"    border: 1px solid #ccc;\n"
"    border-radius: 3px;\n"
"}\n"
"\n"
"QLabel:hover {\n"
"    background-color: #ddd;\n"
"}\n"
"")
        self.Cuda_label.setObjectName("Cuda_label")
        self.gridLayout.addWidget(self.Cuda_label, 6, 0, 1, 1)
        self.Cuda = QtWidgets.QLineEdit(self.layoutWidget)
        self.Cuda.setStyleSheet("/* Style for QLineEdit */\n"
"QLineEdit {\n"
"    font-size: 20px;\n"
"    padding: 5px;\n"
"    border: 1px solid #ccc;\n"
"    border-radius: 3px;\n"
"}\n"
"\n"
"QLineEdit:focus {\n"
"    border-color: #007BFF;\n"
"}")
        self.Cuda.setObjectName("Cuda")
        self.gridLayout.addWidget(self.Cuda, 6, 1, 1, 1)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1226, 23))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)
        MainWindow.setTabOrder(self.Start_train_button, self.Backbone)
        MainWindow.setTabOrder(self.Backbone, self.Batch_size)
        MainWindow.setTabOrder(self.Batch_size, self.Num_workers)
        MainWindow.setTabOrder(self.Num_workers, self.Epochs)
        MainWindow.setTabOrder(self.Epochs, self.Num_class)
        MainWindow.setTabOrder(self.Num_class, self.Frac)
        MainWindow.setTabOrder(self.Frac, self.Cuda)
        MainWindow.setTabOrder(self.Cuda, self.Pretrain)
        MainWindow.setTabOrder(self.Pretrain, self.Dataset)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Pseudo-algorithm"))
        self.Start_train_button.setText(_translate("MainWindow", "开始"))
        self.Stop_train_button.setText(_translate("MainWindow", "停止"))
        self.label.setText(_translate("MainWindow", "Copyright ©  Pengfei Deng"))
        self.label_2.setText(_translate("MainWindow", "训练过程日志"))
        self.Epochs_label.setText(_translate("MainWindow", "Epochs:"))
        self.Frac_label.setText(_translate("MainWindow", "Frac:"))
        self.Frac.setPlaceholderText(_translate("MainWindow", "0~1"))
        self.Num_workers_label.setText(_translate("MainWindow", "Num-workers:"))
        self.Num_class.setPlaceholderText(_translate("MainWindow", "分类个数"))
        self.Pretrain_label.setText(_translate("MainWindow", "Pretrain:"))
        self.Dataset.setPlaceholderText(_translate("MainWindow", "数据集路径"))
        self.Backbone.setItemText(0, _translate("MainWindow", "inception_v3"))
        self.Backbone.setItemText(1, _translate("MainWindow", "vgg16"))
        self.Backbone.setItemText(2, _translate("MainWindow", "vgg19"))
        self.Backbone.setItemText(3, _translate("MainWindow", "desnet161"))
        self.Backbone.setItemText(4, _translate("MainWindow", "resnet101"))
        self.Backbone.setItemText(5, _translate("MainWindow", "resnet50"))
        self.Backbone.setItemText(6, _translate("MainWindow", "mobilenet"))
        self.Backbone.setItemText(7, _translate("MainWindow", "se_resnet50"))
        self.Num_workers.setPlaceholderText(_translate("MainWindow", "线程数"))
        self.Epochs.setPlaceholderText(_translate("MainWindow", "训练轮数"))
        self.Pretrain.setPlaceholderText(_translate("MainWindow", "True/False"))
        self.Dataset_label.setText(_translate("MainWindow", "Dataset:"))
        self.Batch_size.setPlaceholderText(_translate("MainWindow", "batch大小"))
        self.Batch_size_label.setText(_translate("MainWindow", "Batch-size:"))
        self.Backbone_label.setText(_translate("MainWindow", "Backbone:"))
        self.Num_class_label.setText(_translate("MainWindow", "Num-class:"))
        self.Cuda_label.setText(_translate("MainWindow", "Cuda:"))
        self.Cuda.setPlaceholderText(_translate("MainWindow", "0,1,2.."))