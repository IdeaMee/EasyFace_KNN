import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

K = None   # ใส่เลข k เอง เช่น K = 3 | None = ทดสอบทุก k แล้วเลือกให้

def knn(d, y, k):
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
X = np.array(X, dtype=float)
y = np.array(y)

if K is None:
    D = np.array([np.sum((X - x) ** 2, axis=1) for x in X])
    np.fill_diagonal(D, np.inf)

    def error(k):
        return np.mean([knn(D[i], y, k) != y[i] for i in range(len(X))])

    max_k = np.unique(y, return_counts=True)[1].min()
    errors = []
    for k in range(1, max_k + 1):
        errors.append(error(k))
        print("k =", k, "error =", errors[-1])
    best_k = np.argmin(errors) + 1
    print("best k =", best_k)
else:
    best_k = K
    print("using k =", best_k)

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
while True:
    ret, frame = cap.read()
    cv2.rectangle(frame, (250,120), (390,300), (0,0,255), 2)
    face = cv2.cvtColor(frame[120:300, 250:390, :], cv2.COLOR_BGR2GRAY)
    z = face.flatten()
    name = knn(np.sum((X - z) ** 2, axis=1), y, best_k)
    cv2.putText(frame, name, (250, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
    cv2.imshow('frame', frame)
    cv2.waitKey(1)
