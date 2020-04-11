import numpy as np
import os
import pandas as pd
from scipy.stats.mstats import gmean
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage,optimal_leaf_ordering,leaves_list
import matplotlib.pyplot as plt
import sys

def data_clean(face_data, cast_list):
	m = np.zeros(len(face_data))
	for name in cast_list:
		inds = np.argwhere(face_data[:,0]==name)
		inds = np.reshape(inds, [inds.shape[0],])
		m[inds] = 1
	mask = np.ma.make_mask(m) #if face_data[:,0] is present in cast_list, true,
	face_data = face_data[mask]	 	
	
	names = face_data[:,0]
	unique_names = np.unique(names)
	#you can remove names frm castlist that don't have any tracks
	cast_list = np.intersect1d(unique_names, cast_list)
	
	return face_data, cast_list

def get_cooccurence_info(block_times, ol_cast_list, ol_face_data):
	
	face_data, cast_list = data_clean(ol_face_data, ol_cast_list)

	n_scenes = block_times.shape[0]
	n_face_tracks = face_data.shape[0]
	start_end_times = face_data[:,1:3].astype(np.float)
	track_lengths = face_data[:,3].astype(np.int)
	names = face_data[:,0]
	
	index_names = {}
	for i in range(len(cast_list)):
		index_names[cast_list[i]] = i
	num_cast = len(cast_list)

	#Frame Counts
	print('Binning facetracks into '+str(n_scenes)+' scene blocks')



def get_data(out_folder, movie_name):
	
	###############PART 1: BLOCK TIMES

	block_file = os.path.join('PyData',movie_name+'_scene_bounds_neonSG.npy')
	
	if os.path.exists(block_file) and os.path.isfile(block_file):
		block_times = np.load(block_file)
	else:
		scene_bounds = np.load(os.path.join(out_folder,movie_name+'_scene_bounds.npy'))
		shot_bounds = np.load(os.path.join(out_folder,movie_name+'_shots_arr.npy'))
		fps = 25.000 #Can change it 
		n_scenes = scene_bounds.shape[0]
		block_times = np.zeros([n_scenes,2])
		for i in range(n_scenes):
			block_times[i,0] = float(shot_bounds[scene_bounds[i,0]-1,0])/fps
			block_times[i,1] = float(shot_bounds[scene_bounds[i,1]-1,1])/fps

	##############PART 2: CASTLIST
	DataFold = './PyData'
	cast_file = open(os.path.join(DataFold,movie_name+'_castlist.dat'),'r')	
	cast_list = cast_file.read().split(' \n')
	cast_file.close()
	cast_list = cast_list[:-1] #had empty line at last
	
	##############PART 3:FACE DATA
	face_file = open(os.path.join(DataFold,movie_name+'_facetracks.dat'),'r')
	face_lines = face_file.readlines()
	face_data = []
	for l in face_lines:
		n,s,e,tl = l.split()
		face_data.append([n,float(s),float(e),int(tl)])
	face_data = np.array(face_data)
	
	return block_times, cast_list, face_data


if __name__ == '__main__':

	out_folder = '../../outputs'
	movie_name = 'BBT_S1_ep1'

	block_times, cast_list, face_data = get_data(out_folder, movie_name)
	n_scenes = block_times.shape[0]
	np.save('./PyData/BBT_S1_ep1_block_times.npy',block_times)

	##############################################
	get_cooccurence_info(block_times, cast_list, face_data)
