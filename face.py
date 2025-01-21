import cv2
from cvzone.ClassificationModule import Classifier
import numpy as np
import math

cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

classifier = Classifier(
    "C:\\Users\\SyedShouzabHasan\\Desktop\\fc\\model\\keras_model.h5",
    "C:\\Users\\SyedShouzabHasan\\Desktop\\fc\\model\\labels.txt"
)

offset = 20
imgSize = 300
labels = ["Shouzab", "ahad", "sameer", "haider"]

while True:
    success, img = cap.read()
    if not success:
        break

    imgOutput = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
        try:
            imgCrop = img[max(0, y-offset):min(y + h + offset, img.shape[0]), max(0, x-offset):min(x + w + offset, img.shape[1])]
            imgCropShape = imgCrop.shape

            aspectRatio = h / w

            if aspectRatio > 1:
                k = imgSize / h
                wCal = math.ceil(k * w)
                imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                wGap = math.ceil((imgSize - wCal) / 2)
                imgWhite[:, wGap: wCal + wGap] = imgResize
            else:
                k = imgSize / w
                hCal = math.ceil(k * h)
                imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                hGap = math.ceil((imgSize - hCal) / 2)
                imgWhite[hGap: hCal + hGap, :] = imgResize

            prediction, index = classifier.getPrediction(imgWhite, draw=False)
            cv2.rectangle(imgOutput, (x - offset, y - offset - 70), (x + 100, y - offset), (0, 255, 0), cv2.FILLED)
            cv2.putText(imgOutput, labels[index], (x, y - 30), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 0, 0), 2)
            cv2.rectangle(imgOutput, (x - offset, y - offset), (x + w + offset, y + h + offset), (0, 255, 0), 4)

        except Exception as e:
            print("Error processing face region:", e)

    cv2.imshow("Face Recognition", imgOutput)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()