import cv2
from keras.models import load_model
import numpy as np
from className import class_name

def class_name(class_id):
    if class_id==0:
        name='anger'
        return name
    elif class_id==1:
        name='happy'
        return name
    elif class_id==2:
        name='neutral'
        return name
    elif class_id==3:
        name='sadness'
        return name
    elif class_id==4:
        name='surprise'
        return name
    else:
        name = 'unknown'
        return name

model= load_model('emotion_detector_v2.h5')

face_cascade = cv2.CascadeClassifier('./Data/haarcascades/haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)
while(True):
    _,frame = cap.read()
    gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_img, 1.1, 10)

    for (x, y, w, h) in faces:
        if w != 0:
            only_face = frame[y:y + h, x:x + w]
            only_face_resized = cv2.resize(only_face,(64,64))
            only_face_gray = cv2.cvtColor(only_face_resized,cv2.COLOR_BGR2GRAY)
            img=np.reshape(only_face_gray,[1,64,64,1])
            result = model.predict_classes(img)
            text = class_name(result)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(frame,text,(x,y-20),cv2.FONT_HERSHEY_COMPLEX,0.8,(0,255,0),2)
    cv2.imshow('img', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()