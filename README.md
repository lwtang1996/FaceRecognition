# FaceRecgonition
This is a facerecgonition system.

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
<img src = "E:/1.jpg">
