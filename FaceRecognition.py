import os,dlib,numpy,cv2

predictor_path = 'shape_predictor_68_face_landmarks.dat'
face_rc_model_path = 'dlib_face_recognition_resnet_model_v1.dat'
face_folder_path = 'E:\pyproject\candidate_faces'
#face_folder_path = 'E:\candidate_face'
test_img_path = 'test.jpg'
#test_img_path = 'E:\\test.jpg'


# 读取人脸集、人脸标签
def read_data(path):
    try:
        pic_name_list = os.listdir(path)
        pic_list = []
        for i in pic_name_list:
            whole_path = os.path.join(path, i)
            img = cv2.imread(whole_path)
            pic_list.append(img)
    except IOError:
        print('read error')
        return False
    else:
        print('read successfully')
        return pic_name_list, pic_list

# 人脸检测器
detector = dlib.get_frontal_face_detector()

# 关键点检测器
feature_point = dlib.shape_predictor(predictor_path)

# 人脸参数模型
feature_model = dlib.face_recognition_model_v1(face_rc_model_path)

# 候选人特征向量列表
descriptors = []

if __name__ == '__main__':
    name_list, pic_list = read_data(face_folder_path)
    num = 1
    for i in pic_list:
        # 人脸检测
        dets = detector(i, 1)

        for k, d in enumerate(dets):
            # 关键点检测
            shape = feature_point(i, d)

            # 提取特征，128维
            face_feature = feature_model.compute_face_descriptor(i, shape)
            # print(type(face_feature))

            v = numpy.array(face_feature)
            # print(type(v))
            descriptors.append(v)
            # print(type(descriptors))
            print('人脸特征提取，第 %d 个人' % num)
            num += 1
    # 特征向量列表存入文件
    numpy.save('vectors.npy', descriptors)

    '''
    对单张人脸进行识别
    '''
    test_img = cv2.imread(test_img_path)
    dets = detector(test_img, 1)
    for k, d in enumerate(dets):
        shape = feature_point(test_img, d)
        test_feature = feature_model.compute_face_descriptor(test_img, shape)
        test_feature = numpy.array(test_feature)

dist = []
count = 0
for i in descriptors:
    dist_ = numpy.linalg.norm(i-test_feature)
    print('%s : %f' % (name_list[count], dist_))
    dist.append(dist_)
    count += 1

min_dist = numpy.argmin(dist)
result = name_list[min_dist][:-4]
print(result)








