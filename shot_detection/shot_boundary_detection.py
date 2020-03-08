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
	
	if pixel_wise == True:
		diff = np.abs(curr - prev)
		diff = diff.ravel()
		diff = np.linalg.norm(diff, ord = norm_order)
		diff = diff / rows*cols
		return diff
	
	#Instead of pixel wise, can divide into blocks. Found better results this way. 
	#Computed shots are closer to the real shots this way. 
	block_size = 16
	search_space = 16
	diff = 0

	#For each block. A=starting for that block.	, B= All blocks from [-16,16] from that point.
	for i0 in range(0,rows,16):
		for j0 in range(0,cols,16):
			min_diff = float("inf")
			start_i = max(i0-search_space,0)
			start_j = max(j0-search_space,0)
			
			end_i = min(i0+search_space+1,rows-block_size)
			end_j = min(j0+search_space+1,cols-block_size)
			
			for i in range(start_i,end_i):
				for j in range(start_j,end_j):
					#Diff between A[i0:i0+block_size,j0:j0+block_size] and 
					A = curr[i0:i0+block_size,j0:j0+block_size]
					B = prev[i:i+block_size,j:j+block_size]
					d = np.abs(A-B)
					d = np.linalg.norm(d.ravel(), ord = norm_order)
					min_diff = min(min_diff, d)
			diff = diff + min_diff
	diff = diff / rows*cols
	return diff

def get_DFD_array(input_video, out_folder):

	movie_name = input_video.split('/')[-1]	
	movie_name = movie_name.split('.')[-2]
	dfd_path = os.path.join(out_folder,movie_name+'_dfd.npy')
	print("Movie Name:", movie_name)
	#Load from cache if already exists.	
	if os.path.exists(dfd_path) and os.path.isfile(dfd_path):
		DFD_list = np.load(dfd_path)
		return DFD_list

	inp = cv2.VideoCapture(input_video)
	img_name = movie_name + '_'
	frames_path = os.path.join(out_folder,'frames')
	
	save_frames = True			 
	req_frames = int(inp.get(cv2.CAP_PROP_FRAME_COUNT))
	if os.path.exists(frames_path):
		existing_frames = len(glob.glob(os.path.join(frames_path,img_name+'*')))

		if req_frames == existing_frames:	
			save_frames = False
	else:
		os.mkdir(frames_path)

	save_frames = False
	
	#Store images based on Zero-Based Indexing
	j = 0
	ret,current = inp.read()
	if save_frames == True:
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
		if save_frames == True:
			cv2.imwrite(os.path.join(frames_path,img_name +str(j).zfill(6)+'.jpg'),current)
		j=j+1

		diff = compute_dfd(previous,current)
		DFD_list.append(diff)	
	inp.release()
	
	DFD_list = np.array(DFD_list)
	np.save(dfd_path,DFD_list)
	return DFD_list	

parser = argparse.ArgumentParser(description="Shot Boundary Detection and Representation")

parser.add_argument('--inp_file', type=str, help='Path of Input video file', required=True)
parser.add_argument('--out_folder', type=str, help='NAME of folder to store the output files',required=True)

if __name__ == '__main__' :

	args = parser.parse_args()
	movie_path = args.inp_file
	folder_path = os.path.join(os.getcwd(),args.out_folder)

	print(folder_path)
	assert os.path.exists(movie_path), 'No file exists at '+ movie_path+ '.'

	if not os.path.exists(folder_path):
		os.mkdir(folder_path)
	assert os.path.isdir(folder_path),folder_path+ ' already exists and is not a directory. Please specify a different name.'

	print('Getting DFD array ... ')
	DFD_list = get_DFD_array(movie_path,folder_path)	
	print('Getting Shot Boundaries array ... ')
	shot_bounds = shots_arr_from_DFD(DFD_list,folder_path, movie_path)	
