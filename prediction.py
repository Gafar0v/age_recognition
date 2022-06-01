import tensorflow
from tensorflow.keras.preprocessing import image
import numpy as np
from tensorflow.keras.models import load_model
import os
import time
import cv2



path = 'face.jpg'
face_cascade_db = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)
model = load_model('resnet.h5')
t = time.time()
while (cap.isOpened()):

    success, frame = cap.read()
    img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade_db.detectMultiScale(img_gray, 1.1, 19)

    if time.time() - t >= 1.5:
        t = time.time()
        for (x, y, w, h) in faces:
            crop_img = frame[y:y + h, x:x + w]
            cv2.imwrite(path, crop_img)
            img = image.load_img(path, target_size=(150, 150))
            img_tensor = image.img_to_array(img)
            img_tensor = np.expand_dims(img_tensor, axis=0)
            img_tensor /= 255.
            pred = model.predict(img_tensor)
            global predicted_age
            predicted_age = np.mean(list(pred)[0])
            os.remove(path=path)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, int(y)), (x + w, int((y + h))), (0, 255, 0), 2)
        cv2.putText(frame, str(predicted_age), (x, y - 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 2)

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xff == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
