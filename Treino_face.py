import os
from posixpath import join   
import cv2 as cv
import numpy as np

people = ['Cristiano Ronaldo', 'Messi', 'Neymar']
dir = r'C:\Desenvolvimento\Face'
haar_cascade = cv.CascadeClassifier('haar_face.xml')


features = []
labels = []

def training():
    for person in people:
        path = os.path.join(dir, person)
        label = people.index(person)

        for img in os.listdir(path):
            img_path = os.path.join(path, img)

            img_array = cv.imread(img_path)
            gray = cv.cvtColor(img_array, cv.COLOR_BGR2GRAY)

            faces_rect = haar_cascade.detectMultiScale(gray, 1.1, 4)  

            for(x,y,w,h) in faces_rect:
                face_rois = gray[y:y+h, x:x+w]
                features.append(face_rois)
                labels.append(label)
                
training()

features = np.array(features, dtype='object')
labels = np.array(labels)

face_recognizer = cv.face.LBPHFaceRecognizer_create()

face_recognizer.train(features, labels)
face_recognizer.save('face_trained.yml')
np.save('features.npy', features)
np.save('labels.npy', labels)



            


            



