import cv2
import os
import time
import uuid

IMAGES_PATH = 'Tensorflow/workspace/images/collectedimages' #сделать путь

labels = ['hello', 'thanks', 'yes', 'no', 'iloveyou'] #сделать нужное кол-во меток
number_imgs = 15

for label in labels:
    os.mkdir('Tensorflow\workspace\images\collectedimages\\' + label)
    cap = cv2.VideoCapture(0)
    print('colleting images for {}'.format(label))
    time.sleep(5)
    for imgrum in range(number_imgs):
        succes, img = cap.read()
        imgname = os.path.join(IMAGES_PATH, label, label+'.'+'{}jpg'.format(str(uuid.uuid1())))
        cv2.imwrite(imgname, img)
        cv2.imshow('Image', img)
        time.sleep(2)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()