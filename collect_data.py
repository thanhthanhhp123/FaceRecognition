import cv2
import os
import time

start = 0
name = 'khai'
i = 0
cap = cv2.VideoCapture(0)
# cap.set(1280)
# cap.set(640)
try:
    os.mkdir('Datasets/'+name)
except:
    print('Cannot create a file when that file already exists')
while True:
    ret, frame = cap.read()
    #sleep 500ms
    i+=1
    now = time.time()
    if now - start >= 2:
        start = now
        cv2.imwrite('Datasets/'+name+f'/image{i}.jpg', frame)
        cv2.imshow('Capture', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
cap.release()
cv2.destroyAllWindows()