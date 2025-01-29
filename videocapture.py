import numpy as np
import cv2 as cv
import datetime
from skimage import data,filters
 
cap = cv.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()

# We need to set resolutions. 
# so, convert them from float to integer. 
frame_width = int(cap.get(3)) 
frame_height = int(cap.get(4)) 
   
size = (frame_width, frame_height) 
   
fourcc = cv.VideoWriter_fourcc(*'XVID')
out = cv.VideoWriter('output.avi', fourcc, 20.0, (640, 480))
#result = cv.VideoWriter('test.mp4', cv.VideoWriter_fourcc('m','p','4', 'v'), 10, size)
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
 
    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    # Our operations on the frame come here
    #gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    # describe the font type
    font = cv.FONT_ITALIC
   
    # write current Date & Time on each frame
    date_time = str(datetime.datetime.now())
   
    # write the date time in the video frame
    cv.putText(frame,date_time,(10,100), font, 1,(0,0,255),2,cv.LINE_AA)
    # Display the resulting frame
    cv.imshow('frame', frame)
    if cv.waitKey(1) == ord('q'):
        break
cap.release()
out.release() 

cv.destroyAllWindows()