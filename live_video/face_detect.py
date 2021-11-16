"""FaceDetect holds methods to detect the face from a picture or video stream."""

import cv2
import numpy as np


class FaceDetect():
	def __init__(self):
		self.faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
		self.faces = [] # faces in form of x, y, width, height
		self.shapeOfLastDetectedImg = (0, 0)

	# Public method, lets you use whatever implmentation you want. 
	def getNewFrame(self, img, timeStep=0):
		"""Gets the new frame to show on the screen. Called by external classes, etc."""
		# Always run detection of img's shape changed. Otherwise we might get
		# an index out of bounds exception for when the old faces run off the edges 
			# of the new frame.
		if timeStep % 20 == 0 or np.array(img).shape != self.shapeOfLastDetectedImg:
			self._get_faces_with_haar(img)
			return self._draw_faces(img)
		return self._draw_faces(img)

	def _draw_faces(self, img):
		"""Draws a white box for every face in the self.faces array"""
		if self.faces is None:
			return img
        
        # Make sure it works for images with 3 or 4 channels.
		numChannels = img.shape[-1]
		for face in self.faces:
			x,y,w,h = face
			img[y: y + h, x] = [255] * numChannels # Left off here, need to draw the actual bounding boxes. 
			img[y: y + h, x + w] = [255] * numChannels# Left off here, need to draw the actual bounding boxes. 
			img[y, x:x + w] = [255] * numChannels
			img[y + h, x:x + w] = [255] * numChannels
		return img

	def _get_faces_with_haar(self, img):
		"""Uses haar cascades to detect faces and save them in the self.faces array"""
		if img is None:
			return 

		img = np.array(img)

		# Detect faces
		self.faces = self.faceCascade.detectMultiScale(img, minNeighbors=3, minSize=(int(img.shape[0]/10), int(img.shape[0]/10)))
		self.shapeOfLastDetectedImg = img.shape

		# Print so we know what is happening
		if len(self.faces) > 1:
			print("faces greater than 1 ")
		elif len(self.faces) == 0:
			print("no face found ")


