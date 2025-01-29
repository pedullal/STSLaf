#importing libraries 
import cv2 as cv
import numpy as np 
import os 
import time
import cvzone
from flirpy.camera.lepton import Lepton

#choosing camera
#cap = Lepton()
cap = cv.VideoCapture(0)

def BackgroundDeletion(frame): #delete background from the video
	T, threshold = cv.threshold(frame, 150, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)
	return threshold

def SimpleBlob(frame):
	contours, hierarchy = cv.findContours(frame, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
	maxContour = 0
	secondContour = 0
	for contour in contours:
		contourSize = cv.contourArea(contour)
		if contourSize > maxContour and contourSize > 100:
			maxContour = contourSize
			maxContourData = contour
	mask = np.zeros_like(frame)
	cv.fillPoly(mask,[maxContourData],1)
	
	M = cv.moments(mask)
	cX = int(M["m10"] / M["m00"])
	cY = int(M["m01"] / M["m00"])
	
	noBGcolor = cv.cvtColor(noBG, cv.COLOR_GRAY2BGR)
	simpleblob = cv.circle(noBGcolor, (cX, cY), 10, (255, 0, 0), 2)
	return simpleblob
	

def select_roi(image):
	roi_points = []
	def mouse_callback(event, x, y, flags, param):
		if event == cv.EVENT_LBUTTONDOWN:
			roi_points.append((x, y))
			cv.circle(image, (x, y), 5, (0, 255, 0), -1)
			cv.imshow("Select ROI", image)
	cv.imshow("Select ROI", image)
	cv.setMouseCallback("Select ROI", mouse_callback)
	cv.waitKey(0)
	cv.destroyAllWindows()

	
	if len(roi_points) > 2:
		roi_points = np.array(roi_points)
		mask = np.zeros_like(image)
		cv.fillPoly(mask, [roi_points], (255, 255, 255))
		return mask
	else:
		mask = np.ones_like(image)
		return mask
FirstRunCheck = True

while True:
	ret, frame = cap.read()
	#frame1 = cap.grab().astype(np.float32)
	#frame = 255*(frame1 - frame1.min())/(frame1.max()-frame1.min())
	if FirstRunCheck is True and frame is not None:
		mask = select_roi(frame)
		FirstRunCheck = False
	if FirstRunCheck is False:
		roi = cv.bitwise_and(frame, mask)
		im_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

		if not ret:
			print("Can't receive frame (stream end?). Exiting ...")
			break
	
		noBG = BackgroundDeletion(im_gray)
		simpleblob = SimpleBlob(noBG)
		
		noBGcopy = cv.resize(noBG, (900,500))
		blobcopy =  cv.resize(simpleblob, (900,500))
		roicopy = cv.resize(roi, (900,500))
		cv.imshow('no background', noBGcopy)
		cv.imshow('blob detection', blobcopy)
		cv.imshow('no change', roicopy)
	#print(time.time)
		keyboard = cv.waitKey(30)
		if keyboard == 'q' or keyboard == 27:
			break


#def ChooseBoundaries(): #will have the user choose where the boundaries of the track are


#def CheckForCrash(): #implement FSM


