import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

def knn(X, y, z, k=5):
    d = np.sum((X - z) ** 2, axis=1)
    idx = np.argsort(d)[:k]
    cls, vote = np.unique(y[idx], return_counts=True)
    return cls[np.argmax(vote)]

y = []
X = []
for f in os.listdir():
    if not os.path.isfile(f) and not f.startswith("."):
        for i in os.listdir(f):
            if i.endswith(".jpg"):
                x = plt.imread(f+"/"+i)
                X.append(x.flatten())
                y.append(f)
X = np.array(X)
y = np.array(y)
print(X.shape)
print(y)

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
while True:
    ret, frame = cap.read()
    cv2.rectangle(frame, (250,120), (390,300), (0,0,255), 2)
    face = cv2.cvtColor(frame[120:300, 250:390, :], cv2.COLOR_BGR2GRAY)
    z = face.flatten()
    name = knn(X, y, z)
    print(name)
    cv2.putText(frame, name, (250, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
    cv2.imshow('frame', frame)
    cv2.waitKey(1)