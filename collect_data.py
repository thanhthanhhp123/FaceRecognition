import cv2
import os
import time

start = 0
name = 'thanh'
i = 0
cap = cv2.VideoCapture(1)
# cap.set(1280)
# cap.set(640)
try:
    os.mkdir('Datasets/'+name)
except:
    print('Cannot create a file when that file already exists')
while True:
    ret, frame = cap.read()
    #sleep 500ms
    now = time.time()
    if(now - start >= 500):
        cv2.imwrite('Datasets/'+name+f'/image{i+1}.jpg', frame)
        start = now
    cv2.imshow('Capture', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()