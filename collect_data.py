import cv2
import os
import time

start = 0
name = 'khai'
i = 0
cap = cv2.VideoCapture(0)

cap.set(3, 1280)
cap.set(4, 640)
try:
    os.mkdir('Datasets/'+name)
except:
    print('Cannot create a file when that file already exists')
while True:
    ret, frame = cap.read()
    #sleep 500ms
    now = time.time()
    cv2.imshow('Capture', frame)
    if now - start >= 1:
        start = now
        i += 1
        cv2.imwrite('Datasets/'+name+f'/image{i}.jpg', frame)
        if i == 100:
            break

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()