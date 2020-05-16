import cv2
from keras.models import load_model
import numpy as np
def class_name(class_id):
    if class_id==0:
        name='anger'
        return name
    elif class_id==1:
        name='contempt'
        return name
    elif class_id==2:
        name='disgust'
        return name
    elif class_id==3:
        name='neutral'
        return name
    elif class_id==4:
        name='happy'
        return name
    elif class_id==5:
        name='sadness'
        return name
    elif class_id==6:
        name='surprise'
        return name
    else:
        name = 'unknown'
        return name

face_cascade = cv2.CascadeClassifier('./data/haarcascades/haarcascade_frontalface_default.xml')
model= load_model('emotion_detector_v8.h5')
frame = cv2.imread('0d139a6685b896b1c42e07b719570f91.jpg')
gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray_img, 1.1, 10)

for (x, y, w, h) in faces:
    if w != 0:
        new_img = frame[y:y + h, x:x + w]
        second_try = cv2.resize(new_img, (64, 64))
        second_try_gray = cv2.cvtColor(second_try, cv2.COLOR_BGR2GRAY)
        img = np.reshape(second_try_gray, [1, 64, 64, 1])
        result = model.predict_classes(img)
        text = class_name(result)
    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
    cv2.putText(frame, text, (x, y - 20), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 0), 2)
cv2.imshow('Emotion Detection', frame)
cv2.waitKey(0)
cv2.destroyAllWindows()