import cv2
import numpy as np

from face_detect import * # own py file that contains face detection code
from zero_dce import *


class FrameProcessor():
	def __init__(self):
		self.face_detector = FaceDetect()  # create a face detection instance
		self.zero_dce_model = ZeroDCE()
		self.detect_type = "zero_dce"

	# Public method, lets you use whatever implmentation you want. 
	def getNewFrame(self, img, timeStep=0):
		"""Gets the new frame to show on the screen. Called by external classes, etc."""
		# Always run detection of img's shape changed. Otherwise we might get
		# an index out of bounds exception for when the old faces run off the edges 
			# of the new frame.
		# if timeStep % 20 == 0 or np.array(img).shape != self.shapeOfLastDetectedImg:
		# 	self._get_faces_with_haar(img)
		# 	return self._draw_faces(img)
		if self.detect_type == "haar":
			return self.face_detector.getNewFrame(img, timeStep)
		elif self.detect_type == "zero_dce":
			return self.zero_dce_model.getNewFrame(img, timeStep)
		else:
			print("ERROR: Invalid detect_type")
			return img
