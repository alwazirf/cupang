# -*- coding: utf-8 -*-

################################################################################
# Form generated from reading UI file 'splash_screengSeOzb.ui'
##
# Created by: Qt User Interface Compiler version 5.15.0
##
# WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PyQt5.QtCore import (QCoreApplication, QDate, QDateTime, QMetaObject,
                          QObject, QPoint, QRect, QSize, QTime, QUrl, Qt)
from PyQt5.QtGui import (QBrush, QColor, QConicalGradient, QCursor, QFont,
                         QFontDatabase, QIcon, QKeySequence, QLinearGradient, QPalette, QPainter,
                         QPixmap, QRadialGradient)
from PyQt5.QtWidgets import *


class Ui_SplashScreen(object):
    def setupUi(self, SplashScreen):
        if not SplashScreen.objectName():
            SplashScreen.setObjectName(u"SplashScreen")
        SplashScreen.resize(680, 400)
        self.centralwidget = QWidget(SplashScreen)
        self.centralwidget.setObjectName(u"centralwidget")
        self.verticalLayout = QVBoxLayout(self.centralwidget)
        self.verticalLayout.setSpacing(0)
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.verticalLayout.setContentsMargins(10, 10, 10, 10)
        self.dropShadowFrame = QFrame(self.centralwidget)
        self.dropShadowFrame.setObjectName(u"dropShadowFrame")
        self.dropShadowFrame.setStyleSheet(u"QFrame{\n"
                                           "	\n"
                                           "	background-color: rgb(39, 39, 39);\n"
                                           "	color: rgb(220, 220, 220);\n"
                                           "	border-radius: 20px;\n"
                                           "}\n"
                                           "")
        self.dropShadowFrame.setFrameShape(QFrame.StyledPanel)
        self.dropShadowFrame.setFrameShadow(QFrame.Raised)
        self.label_title = QLabel(self.dropShadowFrame)
        self.label_title.setObjectName(u"label_title")
        self.label_title.setGeometry(QRect(0, 80, 661, 111))
        font = QFont()
        font.setFamily(u"Segoe UI")
        font.setPointSize(40)
        self.label_title.setFont(font)
        self.label_title.setStyleSheet(u"color: rgb(20, 167, 108);")
        self.label_title.setAlignment(Qt.AlignCenter)
        self.label_description = QLabel(self.dropShadowFrame)
        self.label_description.setObjectName(u"label_description")
        self.label_description.setGeometry(QRect(0, 180, 661, 31))
        font1 = QFont()
        font1.setFamily(u"Segoe UI")
        font1.setPointSize(14)
        self.label_description.setFont(font1)
        self.label_description.setStyleSheet(u"color: rgb(128, 128, 128);")
        self.label_description.setAlignment(Qt.AlignCenter)
        self.progressBar = QProgressBar(self.dropShadowFrame)
        self.progressBar.setObjectName(u"progressBar")
        self.progressBar.setGeometry(QRect(70, 240, 500, 23))
        self.progressBar.setStyleSheet(u"QProgressBar {\n"
                                       "\n"
                                       "}")
        self.progressBar.setValue(24)
        self.label_credit = QLabel(self.dropShadowFrame)
        self.label_credit.setObjectName(u"label_credit")
        self.label_credit.setGeometry(QRect(20, 340, 621, 31))
        font1 = QFont()
        font1.setFamily(u"Segoe UI")
        font1.setPointSize(6)
        self.label_credit.setFont(font1)
        self.label_credit.setStyleSheet(u"color: rgb(255, 250, 250);")
        self.label_credit.setAlignment(
            Qt.AlignRight | Qt.AlignTrailing | Qt.AlignVCenter)
        self.label_loading = QLabel(self.dropShadowFrame)
        self.label_loading.setObjectName(u"label_loading")
        self.label_loading.setGeometry(QRect(0, 270, 661, 31))
        font2 = QFont()
        font2.setFamily(u"Segoe UI")
        font2.setPointSize(12)
        self.label_loading.setFont(font2)
        self.label_loading.setStyleSheet(u"color: rgb(255, 250, 250);")
        self.label_loading.setAlignment(Qt.AlignCenter)

        self.verticalLayout.addWidget(self.dropShadowFrame)

        SplashScreen.setCentralWidget(self.centralwidget)

        self.retranslateUi(SplashScreen)

        QMetaObject.connectSlotsByName(SplashScreen)
    # setupUi

    def retranslateUi(self, SplashScreen):
        SplashScreen.setWindowTitle(QCoreApplication.translate(
            "SplashScreen", u"MainWindow", None))
        self.label_title.setText(QCoreApplication.translate(
            "SplashScreen", u"<strong>DIC</strong>APP", None))
        self.label_description.setText(QCoreApplication.translate(
            "SplashScreen", u"<html><head/><body><p><span style=\" font-size:14pt;\">Deteksi Jenis Ikan Cupang</span></p></body></html>", None))
        self.label_credit.setText(QCoreApplication.translate(
            "SplashScreen", u"<html><head/><body><p><span style=\" font-size:9pt; font-weight:600;\">Created:</span><span style=\" font-size:9pt;\">Munawarah</span></p></body></html>", None))
        self.label_loading.setText(QCoreApplication.translate(
            "SplashScreen", u"<html><head/><body><p><span style=\" font-size:12pt;\">Please Wait</span></p></body></html>", None))
    # retranslateUi
