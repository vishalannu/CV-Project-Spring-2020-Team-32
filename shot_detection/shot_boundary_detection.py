import numpy as np
import argparse
import os
import cv2
from copy import deepcopy
import glob
import skimage.morphology as morph


def compute_dfd(prev, curr, pixel_wise = False):
	assert prev.shape == curr.shape, 'Input images must be of same dimensions.'
	norm_order = 1
	rows, cols = prev.shape[:2]
	# ||curr - (prev+	optflow)||
	#Plus divide bby rows*cols
	
	diff = np.abs(curr - prev)
	diff = diff.ravel()
	diff = np.linalg.norm(diff, ord = norm_order)
	diff = diff / rows*cols
	return diff
	

def get_DFD_array(input_video, out_folder):

	movie_name = input_video.split('/')[-1]	
	movie_name = movie_name.split('.')[-2]
	dfd_path = os.path.join(out_folder,movie_name+'_dfd.npy')
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
		previous = deepcopy(current)
		ret, current = inp.read()
		if not ret:
			break
		cv2.imwrite(os.path.join(frames_path,img_name +str(j).zfill(6)+'.jpg'),current)
		j=j+1

		diff = compute_dfd(previous,current)
		DFD_list.append(diff)	
	inp.release()
	
	DFD_list = np.array(DFD_list)
	np.save(dfd_path,DFD_list)
	return DFD_list	


if __name__ == '__main__' :

	movie_path = input('Movie Path')
	folder_path = input('Folder Path')
	
	DFD_list = get_DFD_array(movie_path,folder_path)	
