import numpy as np
import argparse
import os
import cv2
from copy import deepcopy
import glob
import skimage.morphology as morph

def shots_arr_from_DFD(DFD_list, input_video, out_folder):
	
	movie_name = input_video.split('/')[-1]	
	movie_name = movie_name.split('.')[-2]
	shots_path = os.path.join(out_folder,movie_name+'_shots_arr.npy')
	#Load from cache if already exists.
	if os.path.exists(shots_path) and os.path.isfile(shots_path):
		shots_arr = np.load(shots_path)
		return shots_arr

	#Filter the DFD and find thresholded local maxima for shot boundaries.	
	DFD_list = np.expand_dims(DFD_list,0) #size = 1 x N-1 N=no of frames

	filter_length = 11
	# line structure element
	selem = morph.rectangle(1,filter_lengths)
	# modified top-hat
	ocdfd = morph.opening(morph.closing(DFD_list, selem), selem)
	mtdfd = DFD_list - np.minimum(ocdfd, DFD_list)

	diff_threshold = 7
	proximity = 8

	# make peaks sharper by diffing twice
	doublediff_mtdfd = np.diff(np.diff(mtdfd))
	use = np.abs(doublediff_mtdfd.squeeze())

	# find peaks
	sharpchange = np.where(use > diff_threshold)[0].tolist()
	changeloc = []
	k = 0
	while k < len(sharpchange):
		flag = k
		a = sharpchange[k]
		while (sharpchange[k] - a) < proximity:
			k += 1
			if k == len(sharpchange):
				break
		changeloc.append(int(round(np.mean(sharpchange[flag:k]))))

	changeloc = [c + 2 for c in changeloc] 
	# add one to compensate for the double diff
	# add one because k[DFD_list] is 0 based indexing while it should have been DFD(t) = F(x,y,t)- F(MC(x,y),t-1) so t starts at 1.

	shots_arr = [[0,changeloc[0]-1]]
	for i in range(len(changeloc)-1):
		shots_arr.append([changeloc[i],changeloc[i+1]])
	Num_frames = DFD_list.shape[1]+1
	shots_arr.append([changeloc[-1],Num_frames-1])
		
	shots_arr = np.array(shots_arr)
	np.save(shots_path, shots_arr)
	return shots_arr	

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
	
	print('Getting DFD array ... ')
	DFD_list = get_DFD_array(movie_path,folder_path)	
	print('Getting Shot Boundaries array ... ')
	shot_bounds = shots_arr_from_DFD(DFD_list,folder_path, movie_path)	