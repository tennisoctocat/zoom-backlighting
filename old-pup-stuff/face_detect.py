import cv2
import imutils
import numpy as np

# Contains methods to extract the face
faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
faces = [] # Hacky, should make into a class or soemthing instead later

# Public method, lets you use whatever implmentation you want. 
def getNewFrame(img, timeStep):
	if timeStep % 20 == 0:
		_get_img_with_haar(img)
		return _draw_faces(img)
	return _draw_faces(img)

def _draw_faces(img):
	global faces
	for face in faces:
		x,y,w,h = face
		img[y: y + h, x] = [255, 255, 255] # Left off here, need to draw the actual bounding boxes. 
		img[y: y + h, x + w] = [255, 255, 255] # Left off here, need to draw the actual bounding boxes. 
		img[y, x:x + w] = [255, 255, 255]
		img[y + h, x:x + w] = [255, 255, 255]
	return img

def _get_img_with_haar(img):
	global faces 
	img = np.array(img)
	faces = faceCascade.detectMultiScale(img, minNeighbors=3, minSize=(int(img.shape[0]/10), int(img.shape[0]/10)))
	if len(faces) > 1:
			print("faces greater than 1 ")
	if len(faces) != 0:
		x,y,w,h = faces[0]
		# maybe do something here? TODO
		#onlyFaceImg = img[y:y + h, x:x + w]
	else:
		x = 0; y = 0; h, w = img.shape[:2]
		#onlyFaceImg = img
		print("no face found ")
	# draw the x, y, w, h for all bounding boxes found
	# for face in faces:
	# 	x,y,w,h = face
	# 	img[y: y + h, x] = [255, 255, 255] # Left off here, need to draw the actual bounding boxes. 
	# 	img[y: y + h, x + w] = [255, 255, 255] # Left off here, need to draw the actual bounding boxes. 
	# 	img[y, x:x + w] = [255, 255, 255]
	# 	img[y + h, x:x + w] = [255, 255, 255]
	# return img

