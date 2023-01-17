# This script will detect faces via your webcam.
# This detection is perform in real time
# Tested with OpenCV3

import cv2
import time
import numpy as np

cap = cv2.VideoCapture(0)

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# create haar cascade
faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
eyeCascade = cv2.CascadeClassifier("haarcascade_eye.xml")
smileCascade = cv2.CascadeClassifier("haarcascade_smile.xml")
bodyCascade = cv2.CascadeClassifier("haarcascade_upperbody.xml")

while(True):
	# capture frame-by-frame
	ret, frame = cap.read()

	# operations on the frames
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #convert image colour to grayscale
	font = cv2.FONT_HERSHEY_DUPLEX

	#can press q to quit info
	#cv2.putText(image, text, origin, font, fontScale, colour, thickness, lineType)
	cv2.putText(
		frame, "Press 'q' to quit", (40,40),font,
		1, (0,0, 0), 1, cv2.LINE_4)
	
	#  detect faces in the image
	faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.3, #how far user is sitting away from webcam
			minSize=(20, 20), #dimensions of rectangles
            minNeighbors=5,  #amount of poisitive neighbouring rectangles required to pass test
        )
	
	#print("Found {0} faces!".format(len(faces)))

	# draw face, detect and draw eyes, mouth
	for (x, y, w, h) in faces:

		# draw rectangles around each found face
		# cv2.rectangle(image, start point, end_point, colour, thickness)
		# image is where the rectangle is to be drawn, here that is the webcam frame
		cv2.rectangle(frame, (x, y), (x+w, y+h), (117,22,0), 2)

		# scan top half of face for eyes
		top_half = gray[int(y):int(y+h/2), int(x):int(x+w)] #gray top

		eyes = eyeCascade.detectMultiScale(top_half)
		for (ex, ey, ew, eh) in eyes:
			cv2.circle(frame, (int(x+ex+eh/2), int(y+ey+eh/2)), int(ew/10*6), (100, 15, 0), 2)

		#scan bottom half of face for mouth
		bottom_half = gray[int(y+h/2):int(y+h), x:int(x+w)]
		mouth = smileCascade.detectMultiScale(
			bottom_half,
			scaleFactor=1.3,
			minSize=(10,10),
			minNeighbors=3,
		)
		for (mx, my, mw, mh) in mouth:
			cv2.rectangle(frame, (x+mx, int(y+h/2+my)), (x+mx+mw, int(y+h/2+my+mh)), (0, 0, 255), 2)
			break

	# display the resulting frame
	cv2.imshow("Frame", frame)
	#press q to quit program
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

	# refresh every x amount of seconds
	time.sleep(0.5)
# everything done, release the capture
cap.release()
cv2.destroyAllWindows()
