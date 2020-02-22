import numpy as np
import os
import cv2
from copy import deepcopy
import glob

def video_to_frames(input_video, out_folder):

	movie_name = input_video.split('/')[-1]	
	movie_name = movie_name.split('.')[-2]
	print("Movie Name:", movie_name)

	inp = cv2.VideoCapture(input_video)
	img_name = movie_name + '_'
	frames_path = os.path.join(out_folder,'frames')
	
	
	#Store images based on Zero-Based Indexing
	j = 0
	ret,current = inp.read()
	cv2.imwrite(os.path.join(frames_path,img_name +str(j).zfill(6)+'.jpg'),current)

	DFD_list = []
	j = 1
	print()
	while True :
		print("\r Frames "+str(j-1)+"/"+str(req_frames)+"finished so far.",end="") 
		ret, current = inp.read()
		if not ret:
			break
		cv2.imwrite(os.path.join(frames_path,img_name +str(j).zfill(6)+'.jpg'),current)
		j=j+1

	inp.release()
	


if __name__ == '__main__' :

	movie_path = input('Movie Path')
	folder_path = input('Folder Path')
	
	video_to_frames(movie_path,folder_path)	
