import cv2
import numpy as np

faceCascade = cv2.CascadeClassifier("facecascade.xml")
img = cv2.imread("lena.png")
imgGray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

faces = faceCascade.detectMultiScale(imgGray, 1.1,4)

for (x,y,w,h) in faces:
    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)

cv2.imshow("output1",img)
cv2.waitKey(0)
