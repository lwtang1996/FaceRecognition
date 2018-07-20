import cv2, dlib, numpy, os, time

time_start = time.time()

test_img_path = 'tlw.jpg'
predictor_path = 'shape_predictor_68_face_landmarks.dat'
face_rc_model_path = 'dlib_face_recognition_resnet_model_v1.dat'
face_folder_path = 'E:\pyproject\candidate_faces'

detector = dlib.get_frontal_face_detector()
feature_point = dlib.shape_predictor(predictor_path)
feature_model = dlib.face_recognition_model_v1(face_rc_model_path)

descriptors = numpy.load('vectors.npy')
face_cascade = cv2.CascadeClassifier('lbpcascade_frontalface_improved.xml')

test_img = cv2.imread(test_img_path)
# dets = detector(test_img, 1)
gray = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)

time1 = time.time()
dets = face_cascade.detectMultiScale(gray, 1.1, 6)
time2 = time.time()
mark = 0
for (x, y, w, h) in dets:
    mark = 1
    print(x,y,w,h)
    img_show = cv2.rectangle(test_img,(x,y),(x+w,y+h),(0,255,0), 5)
    cv2.imwrite('jjj.jpg',img_show)
    d = dlib.rectangle(numpy.long(x),numpy.long(y),numpy.long(x+w),numpy.long(y+h))
    print(d)
    print(type(d))
    shape = feature_point(test_img, d)
    test_feature = feature_model.compute_face_descriptor(test_img, shape)
    test_feature = numpy.array(test_feature)

# for k, d in enumerate(dets):
#     shape = feature_point(test_img, d)
#     test_feature = feature_model.compute_face_descriptor(test_img, shape)
#     test_feature = numpy.array(test_feature)
if mark != 0:
    dist = []
    count = 0
    name_list = os.listdir(face_folder_path)
    for i in descriptors:
        dist_ = numpy.linalg.norm(i - test_feature)
        print('%s : %f' % (name_list[count], dist_))
        dist.append(dist_)
        count += 1

    min_dist = numpy.argmin(dist)
    print('%s' % name_list[min_dist][:-4])

    time3 = time.time()

    print('人脸检测耗时：', time2 - time1)
    print('人脸识别总耗时：', time3 - time_start)

else:
    print('haven\'t find any people')


