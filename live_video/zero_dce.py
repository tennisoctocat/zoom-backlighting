"""FaceDetect holds methods to detect the face from a picture or video stream."""

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torch.backends.cudnn as cudnn
import torch.optim
import os
import sys
import argparse
import time
import model
import numpy as np
from torchvision import transforms
from PIL import Image
import glob
import time



class ZeroDCE():

	def __init__(self):
		self.params_maps = None
		self.scale_factor = 12
		self.DCE_net = model.enhance_net_nopool(self.scale_factor)#.cuda()
		self.DCE_net.load_state_dict(torch.load('models/snapshots_Zero_DCE++/Epoch99.pth',  map_location=torch.device('cpu')))

	def getNewFrame(self, img, timeStep=0):
		# if timeStep % 20 != 0 and not (self.lastFrame is None):
		#  	return img#self.lastFrame
		with torch.no_grad():
			os.environ['CUDA_VISIBLE_DEVICES']='0'
			#data_lowlight = img#Image.open(image_path)

			data_lowlight = (np.asarray(img)/255.0)

			# Added to account for the alpha channel as well as black and white images
			if len(data_lowlight.shape) < 3:
				print("Less than 3 channels, not doing inference.")
				return

			if data_lowlight.shape[2] > 3:
				data_lowlight = data_lowlight[:, :, :3]


			data_lowlight = torch.from_numpy(data_lowlight).float()

			h=(data_lowlight.shape[0]//self.scale_factor)*self.scale_factor
			w=(data_lowlight.shape[1]//self.scale_factor)*self.scale_factor
			data_lowlight = data_lowlight[0:h,0:w,:]
			data_lowlight = data_lowlight.permute(2,0,1)
			data_lowlight = data_lowlight.unsqueeze(0)#cuda().unsqueeze(0)

			#start = time.time()
			# Just use old parameter settings unless we need to refind them.
			if self.params_maps is None:
				enhanced_image,self.params_maps = self.DCE_net(data_lowlight)
			else:
				enhanced_image = self.DCE_net.enhance(data_lowlight, self.params_maps)

			#end_time = (time.time() - start)
			# necessary to get it in the right format for aiortc again
			enhanced_image = np.array(enhanced_image[0].permute(1, 2, 0) * 255, dtype='uint8')

			#print(end_time)

			# start added

			# print("time to just do enhance is ")

			# enhanced_image_2 = DCE_net.enhance(data_lowlight, params_maps)
			# enhance_time = time.time() - start - end_time
			# print(enhance_time)

			# end added 

			#image_path = image_path.replace('test_data','result_Zero_DCE++')

			# result_path = image_path
			# if not os.path.exists(image_path.replace('/'+image_path.split("/")[-1],'')):
			# 	os.makedirs(image_path.replace('/'+image_path.split("/")[-1],''))
			# # import pdb;pdb.set_trace()
			# torchvision.utils.save_image(enhanced_image, result_path)
			# added
			#torchvision.utils.save_image(enhanced_image, result_path.replace('.png','_enhanced_only.png'))
			#print("new img made")
			#print(enhanced_image.shape)
			#self.lastFrame = enhanced_image
			return enhanced_image #end_time

			