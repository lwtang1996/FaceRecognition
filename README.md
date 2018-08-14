# FaceRecognition
This is a facerecognition system.

## 关于
>本工程基于python、opencv、dlib、pyqt，在pycharm下编译通过，在400人规模的人脸识别上表现尚可。由于人脸规模变大时提取每个人脸特征耗时较长，所以将提取特征和识别分为两个阶段。

## 工程目录结构
* pyproject
	* candidate_faces
		* 400个人的登记照(每人一张)
	* face.py
	* FaceRecognition.py
	* main.py
	* shape_predictor_68_face_landmarks.dat
	* dlib_face_recognition_resnet_model_v1.dat
	* vectors.npy(存储的每个人的人脸特征向量)
	* test.jpg(待测人脸照片)

## 使用方法
第一种方式：先运行FaceRecognition.py产生人脸库的特征向量集，再运行face.py直接打印出识别结果。</br>
第二种方式：先运行FaceRecognition.py产生人脸库的特征向量集，再运行main.py在UI界面中进行人脸识别操作。
>第二种方式运行结果如下图：

![](https://github.com/LWTang/FaceRecgonition/raw/master/Screenshots/1.jpg)

## 7.20号更新！
之前的face.py和main.py都是采用dlib的方法进行人脸检测的，速度极慢，更新后main.py中采取opencv的方法来进行人脸检测(原方法注释掉),新的face_opencv.py可代替face.py实现其功能。如今的目录结构为：
* pyproject
	* candidate_faces
		* 400个人的登记照(每人一张)
	* face.py
	* **face_opencv.py**
	* FaceRecognition.py
	* main.py
	* **lbpcascade_frontalface_improved.xml**
	* shape_predictor_68_face_landmarks.dat
	* dlib_face_recognition_resnet_model_v1.dat
	* vectors.npy(存储的每个人的人脸特征向量)
	* test.jpg(待测人脸照片)
	
采用dlib人脸检测和opencv人脸检测的耗时对比如下(两张人脸测试图)：
* test.jpg

*test.jpg* | **opencv** | **dlib**
--- | --- | ---
**人脸检测耗时(s)** | 1.27 | 10.09
**人脸识别总耗时(s)** | 3.71 | 12.47

* tlw.jpg

*tlw.jpg* | **opencv** | **dlib**
--- | --- | ---
**人脸检测耗时(s)** | 1.48 | 9.95
**人脸识别总耗时(s)** | 3.92 | 12.37