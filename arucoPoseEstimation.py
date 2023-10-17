import numpy as np
import cv2
import sys
import time

#Here we define all possible 21 aruco marker types present in the openCV library

#DICT_4*4_50 
#Here,4*4=size of the aruco marker with 4*4 grid of black and white squares.
#50 means the dictionary in which the markers are stored,typically meaning it has 50 possible combinations of markers.
ARUCO_DICT = {
	"DICT_4X4_50": cv2.aruco.DICT_4X4_50, 
	"DICT_4X4_100": cv2.aruco.DICT_4X4_100,
	"DICT_4X4_250": cv2.aruco.DICT_4X4_250,
	"DICT_4X4_1000": cv2.aruco.DICT_4X4_1000,
	"DICT_5X5_50": cv2.aruco.DICT_5X5_50,
	"DICT_5X5_100": cv2.aruco.DICT_5X5_100,
	"DICT_5X5_250": cv2.aruco.DICT_5X5_250,
	"DICT_5X5_1000": cv2.aruco.DICT_5X5_1000,
	"DICT_6X6_50": cv2.aruco.DICT_6X6_50,
	"DICT_6X6_100": cv2.aruco.DICT_6X6_100,
	"DICT_6X6_250": cv2.aruco.DICT_6X6_250,
	"DICT_6X6_1000": cv2.aruco.DICT_6X6_1000,
	"DICT_7X7_50": cv2.aruco.DICT_7X7_50,
	"DICT_7X7_100": cv2.aruco.DICT_7X7_100,
	"DICT_7X7_250": cv2.aruco.DICT_7X7_250,
	"DICT_7X7_1000": cv2.aruco.DICT_7X7_1000,
	"DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL,
	"DICT_APRILTAG_16h5": cv2.aruco.DICT_APRILTAG_16h5,
	"DICT_APRILTAG_25h9": cv2.aruco.DICT_APRILTAG_25h9,
	"DICT_APRILTAG_36h10": cv2.aruco.DICT_APRILTAG_36h10,
	"DICT_APRILTAG_36h11": cv2.aruco.DICT_APRILTAG_36h11
}

def aruco_display(corners, ids, rejected, image):
    
	if len(corners) > 0:
		
		ids = ids.flatten()
		
		for (markerCorner, markerID) in zip(corners, ids):
			
			corners = markerCorner.reshape((4, 2))
			(topLeft, topRight, bottomRight, bottomLeft) = corners
			
			topRight = (int(topRight[0]), int(topRight[1]))
			bottomRight = (int(bottomRight[0]), int(bottomRight[1]))
			bottomLeft = (int(bottomLeft[0]), int(bottomLeft[1]))
			topLeft = (int(topLeft[0]), int(topLeft[1]))

			cv2.line(image, topLeft, topRight, (0, 255, 0), 2)
			cv2.line(image, topRight, bottomRight, (0, 255, 0), 2)
			cv2.line(image, bottomRight, bottomLeft, (0, 255, 0), 2)
			cv2.line(image, bottomLeft, topLeft, (0, 255, 0), 2)
			
			cX = int((topLeft[0] + bottomRight[0]) / 2.0)
			cY = int((topLeft[1] + bottomRight[1]) / 2.0)
			cv2.circle(image, (cX, cY), 4, (0, 0, 255), -1)
			
			cv2.putText(image, str(markerID),(topLeft[0], topLeft[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
			print("[Inference] ArUco marker ID: {}".format(markerID))
			
	return image

#Distortion coefficients and intrinsic paramters are diff for every camera and need to be calibrated using an aruco marker of known length and id.This is done in the folowing two lines.
distortioncoeffs = np.array([0, 0, 0, 0], dtype=np.float32)
intrinsic_parameters = np.array([[800, 0, 320], [0, 800, 240], [0, 0, 1]], dtype=np.float32)

#Selecting the type of aruco markers we'll be showcasing
aruco_type = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_1000)

#Edge length of aruco markers,only for calibration,here for redundancy
sidelength_marker = 0.02

#This turns on the webcam.
cap = cv2.VideoCapture(0)

while (cap.isOpened()):
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    parameters = cv2.aruco.DetectorParameters_create()
    corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(gray, aruco_type, parameters=parameters)

    if corners:
        rotationvecs, translationvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, sidelength_marker, intrinsic_parameters, distortioncoeffs)
        print(f"{rotationvecs} \n")
        for i in range(len(ids)):
            frame = cv2.drawFrameAxes(frame, intrinsic_parameters, distortioncoeffs,rotationvecs[i],translationvecs[i], 0.02)  
    cv2.aruco.drawDetectedMarkers(frame, corners)
    cv2.imshow('frame', cv2.flip(frame, 1))

    if cv2.waitKey(1) & 0xFF == ord('k'):
        break

cap.release()
cv2.destroyAllWindows()
