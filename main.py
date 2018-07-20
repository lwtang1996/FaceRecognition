# -*- coding: utf-8 -*-

import sys, os, numpy, cv2, dlib
from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_Dialog(object):
    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(699, 300)
        self.toolButton = QtWidgets.QToolButton(Dialog)
        self.toolButton.setGeometry(QtCore.QRect(390, 10, 31, 21))
        self.toolButton.setObjectName("toolButton")
        self.pushButton = QtWidgets.QPushButton(Dialog)
        self.pushButton.setGeometry(QtCore.QRect(460, 10, 75, 23))
        self.pushButton.setObjectName("pushButton")
        self.lineEdit = QtWidgets.QLineEdit(Dialog)
        self.lineEdit.setGeometry(QtCore.QRect(130, 10, 251, 20))
        self.lineEdit.setObjectName("lineEdit")
        self.graphicsView = QtWidgets.QGraphicsView(Dialog)
        self.graphicsView.setGeometry(QtCore.QRect(290, 41, 251, 241))
        self.graphicsView.setObjectName("graphicsView")
        self.label = QtWidgets.QLabel(Dialog)
        self.label.setGeometry(QtCore.QRect(290, 41, 251, 241))
        self.label.setObjectName("label")
        self.label.setScaledContents(True)
        self.graphicsView_2 = QtWidgets.QGraphicsView(Dialog)
        self.graphicsView_2.setGeometry(QtCore.QRect(10, 40, 256, 241))
        self.graphicsView_2.setObjectName("graphicsView_2")
        self.label_2 = QtWidgets.QLabel(Dialog)
        self.label_2.setGeometry(QtCore.QRect(13, 41, 251, 241))
        self.label_2.setObjectName("label_2")
        self.label_2.setScaledContents(True)
        self.label_3 = QtWidgets.QLabel(Dialog)
        self.label_3.setGeometry(QtCore.QRect(10, 10, 111, 21))
        font = QtGui.QFont()
        font.setFamily("Agency FB")
        font.setPointSize(11)
        self.label_3.setFont(font)
        self.label_3.setObjectName("label_3")
        self.label_4 = QtWidgets.QLabel(Dialog)
        self.label_4.setGeometry(QtCore.QRect(550, 210, 81, 21))
        font = QtGui.QFont()
        font.setFamily("Agency FB")
        font.setPointSize(11)
        self.label_4.setFont(font)
        self.label_4.setObjectName("label_4")
        self.lineEdit_2 = QtWidgets.QLineEdit(Dialog)
        self.lineEdit_2.setGeometry(QtCore.QRect(550, 240, 141, 20))
        font = QtGui.QFont()
        font.setFamily("Agency FB")
        font.setPointSize(11)
        self.lineEdit_2.setFont(font)
        self.lineEdit_2.setObjectName("lineEdit_2")

        self.retranslateUi(Dialog)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "龟速人脸识别demo"))
        self.toolButton.setText(_translate("Dialog", "..."))
        self.pushButton.setText(_translate("Dialog", "开始识别"))
        self.lineEdit.setText(_translate("Dialog", "E:/"))
        self.label.setText(_translate("Dialog", "识别结果"))
        self.label_2.setText(_translate("Dialog", "待测人脸"))
        self.label_3.setText(_translate("Dialog", "待测人照片路径："))
        self.label_4.setText(_translate("Dialog", "识别结果："))


class Myshow(QtWidgets.QWidget, Ui_Dialog):
    def __init__(self):
        super(Myshow, self).__init__()
        self.setupUi(self)
        self.pushButton.clicked.connect(self.Recognition)
        self.toolButton.clicked.connect(self.ChoosePath)

        self.predictor_path = 'shape_predictor_68_face_landmarks.dat'
        self.face_rc_model_path = 'dlib_face_recognition_resnet_model_v1.dat'
        self.face_folder_path = 'E:\pyproject\candidate_faces'

        self.name_list = os.listdir(self.face_folder_path)
        self.descriptors = numpy.load('vectors.npy')

        # dlib方法检测人脸
        # self.detector = dlib.get_frontal_face_detector()

        # opencv方法检测人脸
        self.face_cascade = cv2.CascadeClassifier('lbpcascade_frontalface_improved.xml')

        self.feature_point = dlib.shape_predictor(self.predictor_path)
        self.feature_model = dlib.face_recognition_model_v1(self.face_rc_model_path)
        # self.dist = []
        self.test_path = 'E:/'

    def ChoosePath(self):
        file_name = QtWidgets.QFileDialog.getOpenFileName(self, "open file dialog", self.test_path, "图片(*.jpg)")
        print(file_name[0])
        self.test_path = file_name[0]
        self.lineEdit.setText(self.test_path)
        self.label_2.setPixmap(QtGui.QPixmap(self.test_path))

        # 清空不相关内容
        self.label.clear()
        self.lineEdit_2.clear()

    def Recognition(self):
        test_img = cv2.imread(self.test_path)

        # dlib方法检测人脸
        # dets = self.detector(test_img, 1)
        # for k, d in enumerate(dets):
        #     shape = self.feature_point(test_img, d)
        #     test_feature = self.feature_model.compute_face_descriptor(test_img, shape)
        #     test_feature = numpy.array(test_feature)

        # opencv方法检测人脸
        gray = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)
        dets = self.face_cascade.detectMultiScale(gray, 1.1, 6)
        mark = 0
        for (x, y, w, h) in dets:
            mark = 1
            d = dlib.rectangle(numpy.long(x),numpy.long(y),numpy.long(x+w),numpy.long(y+h))
            shape = self.feature_point(test_img, d)
            test_feature = self.feature_model.compute_face_descriptor(test_img, shape)
            test_feature = numpy.array(test_feature)

        if mark == 1:
            dist = []
            count = 0
            for i in self.descriptors:
                dist_ = numpy.linalg.norm(i - test_feature)
                print('%s : %f' % (self.name_list[count], dist_))
                dist.append(dist_)
                count += 1

            min_dist = numpy.argmin(dist)
            print('%s' % self.name_list[min_dist][:-4])

            show_img_path = os.path.join(self.face_folder_path, self.name_list[min_dist])
            self.label.setPixmap(QtGui.QPixmap(show_img_path))
            self.lineEdit_2.setText(self.name_list[min_dist][:-4])
        else :
            self.lineEdit_2.setText('haven\'t find any people')


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    w = Myshow()
    w.show()
    sys.exit(app.exec_())
