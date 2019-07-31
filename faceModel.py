import cv2 as cv
import time
import os
import numpy as np
import sklearn.preprocessing as sp
import face_recognition
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler



class FaceRecognition():
    def __init__(self):
        self.savePath = 'image/img/'
        self.train_faces = self.search_faces('image/img/')


    def delByName(self,name):
        directory = os.path.normpath('image/img/' + name)
        for curdir, subdirs, files in os.walk(directory):
            #删掉里面的照片
            for i in files:
                os.remove(curdir+'/'+i)



    def picCap(self,name,sec):
        logo = cv.imread('logo/logo.jpg')
        vc = cv.VideoCapture(0)
        t_end = time.time() + sec
        capturedNum = 0
        #删掉之前的照片
        self.delByName(name)
        print("请将脸放入摄像头范围，稍微动一动")
        cv.namedWindow(name, cv.WINDOW_AUTOSIZE)
        cv.setWindowProperty(name, cv.WND_PROP_FULLSCREEN, cv.WINDOW_FULLSCREEN)
        cv.moveWindow(name, 0, 0)
        while time.time() < t_end and capturedNum<=2:
            frame = vc.read()[1]
            faces = face_recognition.face_locations(frame)
            if faces != ():
                capturedNum += 1
                cv.imwrite(self.savePath + '/' + name + '/' + str(capturedNum) + '.jpg',
                                frame)

                t = str(int(t_end - time.time())) + 's'
                frame = cv.putText(frame,t,(20,100),cv.FONT_HERSHEY_COMPLEX,2,(0,255,0),3)
                cv.imshow(name, frame)
                time.sleep(0.5)
            else:
                t = str(int(t_end - time.time())) + 's'
                frame = cv.putText(frame, t, (20, 100), cv.FONT_HERSHEY_COMPLEX,2, (0, 255, 0), 3)
                cv.imshow(name, frame)
            #按了ESC  33ms检测一次
            cv.waitKey(33)
        #判断采集是否成功 成功截图数是否够。
        directory = os.path.normpath('image/img/' + name)
        for curdir, subdirs, files in os.walk(directory):
            if len(files) > 1:
                print("采集成功")
                cv.imshow(name, logo)
                cv.waitKey(100)
                cv.destroyAllWindows()
                vc.release()
                return True
            else:
                print("采集失败")
                cv.imshow(name, logo)
                cv.waitKey(100)
                cv.destroyAllWindows()
                vc.release()
                return False


    def search_faces(self,directory):
        directory = os.path.normpath(directory)
        faces = {}
        for curdir, subdirs, files in os.walk(directory):
            for jpeg in (file for file in files
                         if file.endswith('.jpg')):
                path = os.path.join(curdir, jpeg)
                label = path.split(os.path.sep)[-2]
                if label not in faces:
                    faces[label] = []
                faces[label].append(path)
        return faces

#方法1 用的face_landmarks SCV
    def trainModel(self):
        #{'hai':[url1,url2]

        self.codec = sp.LabelEncoder()
        self.codec.fit(list(self.train_faces.keys()))
        train_x, train_y = [], []
        for label, filenames in self.train_faces.items():
            for filename in filenames:
                image = face_recognition.load_image_file(filename)
                face_landmarks_list = face_recognition.face_landmarks(image)
                print('label:',label,'filename:',filename,':',face_landmarks_list)
                featureList = []
                for key in face_landmarks_list[0]:
                    for i in face_landmarks_list[0].get(key):
                        a,b = i
                        featureList.append(a)
                        featureList.append(b)

                train_x.append(featureList)
                train_y.append(self.codec.transform([label])[0])

        train_x = np.array(train_x)

        #数据预处理
        self.train_data_scalar = StandardScaler().fit(train_x)
        train_x = self.train_data_scalar.transform(train_x)
        print("standard:",train_x)

        train_y = np.array(train_y)
        #支持向量机分类器
        self.model = SVC(kernel='linear', probability=True)
        self.model.fit(train_x, train_y)

    def predictFace(self,sec):
        vc = cv.VideoCapture(0)

        for i in range(10):
            frame = vc.read()[1]
            cv.waitKey(33)
        #只进行sec秒检测
        t_end = time.time() + sec
        while time.time() < t_end:
            frame = vc.read()[1]
            face = face_recognition.face_locations(frame)
            pred_test_x = []
            if face != ():
                face_landmarks_list = face_recognition.face_landmarks(frame,face)
                featureList = []
                if len(face_landmarks_list) <= 0:
                    continue
                for key in face_landmarks_list[0]:
                    for i in face_landmarks_list[0].get(key):
                        a, b = i
                        featureList.append(a)
                        featureList.append(b)
                pred_test_x.append(featureList)
                pred_test_x = np.array(pred_test_x)

                #数据预处理
                pred_test_x = self.train_data_scalar.transform(pred_test_x)
                print("standard:", pred_test_x)

                pred_code = self.model.predict(pred_test_x)[0]
                pred_test_y = self.codec.inverse_transform([pred_code])

                # 计算置信概率
                probs = self.model.predict_proba(pred_test_x)[0]
                print(probs)
                # resList = self.getMeansByStd(probs)
                break
            # 按了ESC  33ms检测一次
            cv.waitKey(500)
        else:
            #cv.destroyAllWindows()
            vc.release()
            print('vague')
            return 'vague'
        #cv.destroyAllWindows()
        vc.release()
        print(pred_test_y[0])
        return pred_test_y[0]

#方法2 encoding最不准
    # def trainModel(self):
    #     #{'hai':[url1,url2]
    #
    #     self.codec = sp.LabelEncoder()
    #     self.codec.fit(list(self.train_faces.keys()))
    #     train_x, train_y = [], []
    #     for label, filenames in self.train_faces.items():
    #         for filename in filenames:
    #             image = face_recognition.load_image_file(filename)
    #             face_location = face_recognition.face_locations(image)
    #             face_encodings = face_recognition.face_encodings(image,face_location)
    #             print('label:', label, 'filename:', filename, ":", face_encodings[0])
    #             train_x.append(face_encodings[0])
    #             train_y.append(self.codec.transform([label])[0])
    #     train_x = np.array(train_x)
    #     train_y = np.array(train_y)
    #     #支持向量机分类器
    #     self.model = SVC(kernel='linear', probability=True)
    #     self.model.fit(train_x, train_y)
    #
    # def predictFace(self,sec):
    #     vc = cv.VideoCapture(0)
    #
    #     for i in range(10):
    #         frame = vc.read()[1]
    #         cv.waitKey(33)
    #     #只进行sec秒检测
    #     t_end = time.time() + sec
    #     while time.time() < t_end:
    #         frame = vc.read()[1]
    #         face = face_recognition.face_locations(frame)
    #         pred_test_x = []
    #         if face != ():
    #             face_encoding = face_recognition.face_encodings(frame,face)
    #             pred_test_x.append(face_encoding[0])
    #             pred_test_x = np.array(pred_test_x)
    #             pred_code = self.model.predict(pred_test_x)[0]
    #             pred_test_y = self.codec.inverse_transform([pred_code])
    #
    #             # 计算置信概率
    #             probs = self.model.predict_proba(pred_test_x)[0]
    #             print(probs)
    #             # resList = self.getMeansByStd(probs)
    #             break
    #         # 按了ESC  33ms检测一次
    #         cv.waitKey(500)
    #     else:
    #         #cv.destroyAllWindows()
    #         vc.release()
    #         print('vague')
    #         return 'vague'
    #     #cv.destroyAllWindows()
    #     vc.release()
    #     print(pred_test_y[0])
    #     return pred_test_y[0]

#方法3
    # def trainModel(self):
    #     self.faceLib = []
    #     self.labels = []
    #     for subdirs in os.listdir('image/img'):
    #         #sun lin chen
    #         self.faceLib.append(face_recognition.face_encodings(face_recognition.load_image_file('image/img/'+subdirs+'/1.jpg'))[0])
    #         self.labels.append(subdirs)
    #     self.faceLib = np.array(self.faceLib)
    #     self.labels = np.array(self.labels)
    #     print(self.faceLib)
    #
    # def predictFace(self,sec):
    #     vc = cv.VideoCapture(0)
    #
    #     for i in range(10):
    #         frame = vc.read()[1]
    #         cv.waitKey(33)
    #     #只进行sec秒检测
    #     t_end = time.time() + sec
    #     while time.time() < t_end:
    #         frame = vc.read()[1]
    #         face_location = face_recognition.face_locations(frame)
    #         print("fl:",face_location)
    #         face_encoding = face_recognition.face_encodings(frame,face_location)
    #         print('fe:',face_encoding)
    #         #[f t f f f f]  compare_face做了 np.linalg.norm(face_encodings - face_to_compare, axis=1)
    #         if len(face_location)>0 and len(face_encoding)>0:
    #             match = face_recognition.compare_faces(self.faceLib,face_encoding, tolerance=0.2)
    #             print(self.labels[match])
    #             return self.labels[match][0]
    #     else:
    #         return 'vague'

    # def show(self,path,type):
    #     print('选中图片按ESC 关闭窗口')
    #     src = cv.imread(path)
    #     cv.namedWindow(type, cv.WINDOW_AUTOSIZE)
    #     cv.setWindowProperty(type, cv.WND_PROP_FULLSCREEN, cv.WINDOW_FULLSCREEN)
    #     cv.moveWindow(type, 0, 0)
    #     while True:
    #         # print(cv.getWindowProperty(type, cv.WND_PROP_AUTOSIZE))
    #         cv.imshow(type, src)
    #         k = cv.waitKey(33)
    #
    #         if k == 27:
    #             break
    #
    #     logo = cv.imread('logo/logo.jpg')
    #     cv.imshow(type, logo)
    #     cv.waitKey(100)
    #     cv.destroyAllWindows()

if __name__ == '__main__':
    fr = FaceRecognition()
    # while True:
    #     k = input("输入你的姓名: /detective du hai host liu professor   输入q退出")
    #     if k == 'q':
    #         break
    #     else:
    #         fr.picCap(k,20)

    fr.trainModel()
    fr.predictFace(6)



#
#
# import cv2 as cv
# import time
# import os
# import numpy as np
# import sklearn.preprocessing as sp
#
# class FaceRecognition():
#     def __init__(self):
#         # 哈尔级联人脸定位器
#         self.fd = cv.CascadeClassifier('ml_data/haar/face.xml')
#         self.ed = cv.CascadeClassifier('ml_data/haar/eye.xml')
#         self.nd = cv.CascadeClassifier('ml_data/haar/nose.xml')
#         self.savePath = 'image/img/'
#         self.train_faces = self.search_faces('image/img/')
#
#     def delByName(self,name):
#         directory = os.path.normpath('image/img/' + name)
#         for curdir, subdirs, files in os.walk(directory):
#             #删掉里面的照片
#             for i in files:
#                 os.remove(curdir+'/'+i)
#
#
#
#     def picCap(self,name,sec):
#         logo = cv.imread('logo/logo.jpg')
#         vc = cv.VideoCapture(0)
#         t_end = time.time() + sec
#         capturedNum = 0
#         #删掉之前的照片
#         self.delByName(name)
#         print("请将脸放入摄像头范围，稍微动一动，尽量出现更多的彩色圆圈")
#         cv.namedWindow(name, cv.WINDOW_AUTOSIZE)
#         cv.setWindowProperty(name, cv.WND_PROP_FULLSCREEN, cv.WINDOW_FULLSCREEN)
#         cv.moveWindow(name, 0, 0)
#         while time.time() < t_end and capturedNum<=5:
#             frame = vc.read()[1]
#             # 1.3 scaleFactor 表示在前后两次相继的扫描中，搜索窗口的比例系数。默认为1.1即每次搜索窗口依次扩大10%;
#             # 5 minNeighbors 表示构成检测目标的相邻矩形的最小个数(默认为3个)。 如果min_neighbors 为 0, 则函数不做任何操作就返回所有的被检候选矩形框，
#             faces = self.fd.detectMultiScale(frame, 1.3, 6)
#             if faces != ():
#                 capturedNum += 1
#                 cv.imwrite(self.savePath + '/' + name + '/' + str(capturedNum) + '.jpg',
#                                 frame)
#                 for l, t, w, h in faces:
#                     a, b = int(w / 2), int(h / 2)
#                     cv.ellipse(frame, (l + a, t + b),
#                                (a, b), 0, 0, 360,
#                                (255, 0, 255), 2)
#                     face = frame[t:t + h, l:l + w]
#                     eyes = self.ed.detectMultiScale(face, 1.3, 5)
#                     for l, t, w, h in eyes:
#                         a, b = int(w / 2), int(h / 2)
#                         cv.ellipse(face, (l + a, t + b),
#                                    (a, b), 0, 0, 360,
#                                    (0, 255, 0), 2)
#                     noses = self.nd.detectMultiScale(face, 1.3, 5)
#                     for l, t, w, h in noses:
#                         a, b = int(w / 2), int(h / 2)
#                         cv.ellipse(face, (l + a, t + b),
#                                    (a, b), 0, 0, 360,
#                                    (0, 255, 255), 2)
#                     t = str(int(t_end - time.time())) + 's'
#                     frame = cv.putText(frame,t,(20,100),cv.FONT_HERSHEY_COMPLEX,2,(0,255,0),3)
#                     cv.imshow(name, frame)
#                     time.sleep(0.5)
#             else:
#                 t = str(int(t_end - time.time())) + 's'
#                 frame = cv.putText(frame, t, (20, 100), cv.FONT_HERSHEY_COMPLEX,2, (0, 255, 0), 3)
#                 cv.imshow(name, frame)
#             #按了ESC  33ms检测一次
#             cv.waitKey(33)
#         #判断采集是否成功 成功截图数是否够。
#         directory = os.path.normpath('image/img/' + name)
#         for curdir, subdirs, files in os.walk(directory):
#             if len(files) > 3:
#                 print("采集成功")
#                 cv.imshow(name, logo)
#                 cv.waitKey(100)
#                 cv.destroyAllWindows()
#                 vc.release()
#                 return True
#             else:
#                 print("采集失败")
#                 cv.imshow(name, logo)
#                 cv.waitKey(100)
#                 cv.destroyAllWindows()
#                 vc.release()
#                 return False
#
#
#     def search_faces(self,directory):
#         directory = os.path.normpath(directory)
#         faces = {}
#         for curdir, subdirs, files in os.walk(directory):
#             for jpeg in (file for file in files
#                          if file.endswith('.jpg')):
#                 path = os.path.join(curdir, jpeg)
#                 label = path.split(os.path.sep)[-2]
#                 if label not in faces:
#                     faces[label] = []
#                 faces[label].append(path)
#         return faces
#
#     def trainModel(self):
#         #{'hai':[url1,url2]
#
#         self.codec = sp.LabelEncoder()
#         self.codec.fit(list(self.train_faces.keys()))
#         train_x, train_y = [], []
#         for label, filenames in self.train_faces.items():
#             for filename in filenames:
#                 image = cv.imread(filename)
#                 gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
#                 faces = self.fd.detectMultiScale(gray, 1.1, 3,
#                                             minSize=(100, 100))
#                 for l, t, w, h in faces:
#
#                     train_x.append(
#                         gray[t:t + h, l:l + w])
#
#                     train_y.append(
#                         self.codec.transform([label])[0])
#         train_y = np.array(train_y)
#         # 局部二值模式直方图人脸识别分类器
#         self.model = cv.face.LBPHFaceRecognizer_create()
#         self.model.train(train_x, train_y)
#
#
#     def predictFace(self,sec):
#         vc = cv.VideoCapture(0)
#
#         for i in range(10):
#             frame = vc.read()[1]
#             cv.waitKey(33)
#         #只进行sec秒检测
#         t_end = time.time() + sec
#         while time.time() < t_end:
#             frame = vc.read()[1]
#             face = self.fd.detectMultiScale(frame,1.3,5)
#             testFace = None
#             if face != ():
#                 #print(frame)
#                 gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
#                 faces = self.fd.detectMultiScale(gray, 1.1, 2,
#                                                  minSize=(100, 100))
#                 for l, t, w, h in faces:
#                     testFace = gray[t:t + h, l:l + w]
#                 pred_code = self.model.predict(testFace)[0]
#                 pred_test_y = self.codec.inverse_transform([pred_code])
#                 #print(pred_test_y[0])
#                 break
#             # 按了ESC  33ms检测一次
#             cv.waitKey(500)
#         else:
#             #cv.destroyAllWindows()
#             vc.release()
#             return 'vague'
#         #cv.destroyAllWindows()
#         vc.release()
#         return pred_test_y[0]
#
#     def show(self,path,type):
#         print('选中图片按ESC 关闭窗口')
#         src = cv.imread(path)
#         cv.namedWindow(type, cv.WINDOW_AUTOSIZE)
#         cv.setWindowProperty(type, cv.WND_PROP_FULLSCREEN, cv.WINDOW_FULLSCREEN)
#         cv.moveWindow(type, 0, 0)
#         while True:
#             # print(cv.getWindowProperty(type, cv.WND_PROP_AUTOSIZE))
#             cv.imshow(type, src)
#             k = cv.waitKey(33)
#
#             if k == 27:
#                 break
#
#         logo = cv.imread('logo/logo.jpg')
#         cv.imshow(type, logo)
#         cv.waitKey(100)
#         cv.destroyAllWindows()
#
# if __name__ == '__main__':
#     fr = FaceRecognition()
#     while True:
#         k = input("输入你的姓名: /detective du hai host liu professor   输入q退出")
#         if k == 'q':
#             break
#         else:
#             fr.picCap(k,20)
#
#     fr.trainModel()
#     print(fr.predictFace(5))
