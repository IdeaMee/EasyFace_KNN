import cv2

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
while True:
    ret, frame = cap.read()
    cv2.imshow('frame', frame)
    cv2.waitKey(1)