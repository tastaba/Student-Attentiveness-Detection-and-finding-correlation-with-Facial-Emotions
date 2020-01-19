# USAGE
# python motion_detector.py
#for running with a pre-recorded video add --video videos/example.mp4 to parameters in the run configurations.

# import the necessary packages
from imutils.video import VideoStream
import argparse
import datetime
import imutils
import time
import cv2

# load the required trained XML classifiers
# https://github.com/Itseez/opencv/blob/master/
# data/haarcascades/haarcascade_frontalface_default.xml
# Trained XML classifiers describes some features of some
# object we want to detect a cascade function is trained
# from a lot of positive(faces) and negative(non-faces) images.
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
right_face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_profileface.xml")

# /data/haarcascades/haarcascade_eye.xml
# Trained XML file for detecting eyes
# eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
# construct the argument parser and parse the arguments

ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", help="path to the video file")
ap.add_argument("-a", "--min-area", type=int, default=600, help="minimum area size")#minimum area size is the minimum
#area of detected change of activities.
args = vars(ap.parse_args())

# if the video argument is None, then we are reading from webcam
if args.get("video", None) is None:
	vs = VideoStream(src=0).start()
	time.sleep(2.0)

# otherwise, we are reading from a video file
else:
	vs = cv2.VideoCapture(args["video"])

# initialize the first frame in the video stream
firstFrame = None
hasChanged = False
count = 0
frameChangeThresh = 5
minRectArea = 0
crop_img = None

start = time.time()#taking the time of when the first frame was detected.
# loop over the frames of the video
while True:
	# grab the current frame and initialize the occupied/unoccupied
	# text
	frame = vs.read()
	frame = frame if args.get("video", None) is None else frame[1]
	text = "Unoccupied"

	# if the frame could not be grabbed, then we have reached the end
	# of the video
	if frame is None:
		break

	# resize the frame, convert it to grayscale, and blur it
	frame = imutils.resize(frame, width=700)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	gray = cv2.GaussianBlur(gray, (21, 21), 0)

	# Detects faces of different sizes in the input image
	faces = face_cascade.detectMultiScale(gray, 1.3, 5)
	if(len(faces)==0):
		faces = right_face_cascade.detectMultiScale(gray,1.3,5)
		if (len(faces) == 0):
			gray = cv2.flip(gray,1)
			faces = right_face_cascade.detectMultiScale(gray, 1.3, 5)
		if (len(faces) == 0):
			text = "Unoccupied"
	#Finding the face with maximum area among multiple detected faces.
	selectedFace = None
	minRectArea = 0
	for (x, y, w, h) in faces:
		if(w*h*1.25*1.25>minRectArea):
			minRectArea = w * h*1.25*1.25
			selectedFace = (x,y,w,h)

	for (x, y, w, h) in faces:
		if (x,y,w,h)==selectedFace:
			# To draw a bounding box around a face
			cv2.rectangle(frame, (x - int(w*0.35), y- int(h*0.35)), (x + int(w*1.25), y + int(h*1.25)), (255, 255, 0), 2)
			text = "Occupied"
			roi_gray = gray[y:y + h, x:x + w]
			roi_color = frame[y:y + h, x:x + w]
			crop_img = frame[y- int(h*0.35):y + int(h*1.25), x - int(w*0.35):x + int(w*1.25)]

		# Detects eyes of different sizes in the input image
		# eyes = eye_cascade.detectMultiScale(roi_gray)

		# To draw a rectangle in eyes
		# for (ex, ey, ew, eh) in eyes:
		# 	cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 127, 255), 2)

		# Display an image in a window
	#cv2.imshow('img', frame)

	# Wait for Esc key to stop
	k = cv2.waitKey(30) & 0xff
	if k == 27:
		break

# if the first frame is None, initialize it
	if firstFrame is None:
		firstFrame = gray
		continue

	# compute the absolute difference between the current frame and
	# first frame
	frameDelta = cv2.absdiff(firstFrame, gray)
	thresh = cv2.threshold(frameDelta, 5, 255, cv2.THRESH_BINARY)[1]

	# dilate the thresholded image to fill in holes, then find contours
	# on thresholded image
	thresh = cv2.dilate(thresh, None, iterations=2)
	cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
		cv2.CHAIN_APPROX_SIMPLE)
	cnts = imutils.grab_contours(cnts)

	# loop over the contours
	for c in cnts:
		# if the contour is too small, ignore it
		if cv2.contourArea(c) < args["min_area"]:
			continue

		# compute the bounding box for the contour, draw it on the frame,
		# and update the text
		(x, y, w, h) = cv2.boundingRect(c)
		#cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
		#text = "Occupied"
		hasChanged = True
		#firstFrame =  frame;
	if(hasChanged):
		firstFrame = None
		hasChanged = False
		end = time.time()
		if((end-start)>frameChangeThresh):
			cv2.imwrite("frame%d.jpg" % count, crop_img)
			count = count+1
			start = time.time()
	# draw the text and timestamp on the frame
	cv2.putText(frame, "Room Status: {}".format(text), (10, 20),
	cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
	cv2.putText(frame, datetime.datetime.now().strftime("%A %d %B %Y %I:%M:%S%p"),
	(10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)

	# show the frame and record if the user presses a key
	cv2.imshow("Security Feed", frame)
	#cv2.imshow("Thresh", thresh)
	#cv2.imshow("Frame Delta", frameDelta)
	key = cv2.waitKey(1) & 0xFF

	# if the `q` key is pressed, break from the loop
	if key == ord("q"):
		break

# cleanup the camera and close any open windows
vs.stop() if args.get("video", None) is None else vs.release()
cv2.destroyAllWindows()