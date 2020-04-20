import numpy as np
import os
import pandas as pd
from scipy.stats.mstats import gmean
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage,optimal_leaf_ordering,leaves_list
from losses import loss_function
from sg_optimization import storygraph_optimization
import matplotlib.pyplot as plt
import sys
from draw_graph import draw_graph

def graph_initialization(num_cast, n_scenes, lin_cooc, presence):
	#Init the indices xct
	init_indices = np.zeros([num_cast,n_scenes])
	init_method = 'olo_special'
	
	if init_method == 'alphabetical':
		init_indices = np.arange(1,num_cast+1).reshape(num_cast,1)@np.ones([1,n_scenes])
	elif init_method == 'olo' : #based on global co occurence
		v = np.sum(lin_cooc,1)
		D = 1/squareform(v)

		D[np.isinf(D)] = 0
		olo = leaves_list(optimal_leaf_ordering(linkage(D),D))
		order = np.argsort(olo)+1
		init_indices = order.reshape(num_cast,1)@np.ones([1,n_scenes])	
	elif init_method == 'olo_special' : #based on special co occurence XOR based function
		#XOR and presence based dist mat
		dist_mat = np.zeros([num_cast,num_cast])
		olopresence = presence
		for ii in range(num_cast):
			for jj in range(ii+1,num_cast):
				sx1 = np.sum(olopresence[ii,:])	
				sx2 = np.sum(olopresence[jj,:])	
				x1_and_x2 = np.logical_and(olopresence[ii,:],olopresence[jj,:]) 
				x1_xor_x2 = np.logical_xor(olopresence[ii,:],olopresence[jj,:]) 
				
				measure = 1- sum(x1_and_x2)/np.sqrt(sx1*sx2) + 2*sum(x1_xor_x2)/(sx1+sx2)
				dist_mat[ii,jj] = measure
				dist_mat[jj,ii] = measure 			
		
		D = squareform(dist_mat)
		D[np.isinf(D)] = 0
		olo = leaves_list(optimal_leaf_ordering(linkage(D),D))
		order = np.argsort(olo)+1
		init_indices = order.reshape(num_cast,1)@np.ones([1,n_scenes])	
		
	elif init_method == 'olo_random':
		order = np.randon.permutation(np.arange(1,num_cast+1))
		init_indices = order.reshape(num_cast,1)@np.ones([1,n_scenes])	

	return init_indices

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
	frame_counts = np.zeros([num_cast,n_scenes])
	for i in range(n_scenes):
		ind1 = np.argwhere(start_end_times[:,0]>=block_times[i,0])
		ind1 = np.reshape(ind1,[ind1.shape[0],])
		ind2 = np.argwhere(start_end_times[:,1] <= block_times[i,1])
		ind2 = np.reshape(ind2,[ind2.shape[0],])
		tracks_in_interval = np.intersect1d(ind1,ind2)
		names_in_interval = np.take(names,tracks_in_interval)
		tlengths_in_interval = np.take(track_lengths,tracks_in_interval)
		n_inblock = np.unique(names_in_interval) #can have unknown
		for n in n_inblock:
			this_name_idx = np.argwhere(names_in_interval==n)#indexes in names_in_interval where n is present
			#if n == 'howard':
			#print("interesting",this_name_idx)
			ind = index_names[n]
			frame_counts[ind,i] = sum(tlengths_in_interval[this_name_idx])		
	#Presence
	presence = np.where(frame_counts>0, 1, 0)

	#Cooccurence
	cooc_mat = np.zeros([num_cast,num_cast,n_scenes])
	for k in range(n_scenes):
		for i in range(num_cast):
			for j in range(i+1,num_cast):
				cooc_mat[i,j,k] = gmean([frame_counts[i,k],frame_counts[j,k]])
	#LInearize
	u,v = np.triu_indices(num_cast,1)
	lin_cooc = np.zeros([len(u),n_scenes])
	for k in range(n_scenes):
		c = cooc_mat[:,:,k]
		lin_cooc[:,k] = c[u,v]

	#Initialize the graph
	init_indices = graph_initialization(num_cast, n_scenes, lin_cooc,presence)	
	
	#Normalize
	norm_method = 'hysteresis'
	if norm_method == 'each': #normalize each column to 1
		lin_cooc = lin_cooc/np.amax(lin_cooc,axis=0)
	elif norm_method == 'all' : #normalize max value to 1
		lin_cooc = lin_cooc/np.amax(np.amax(lin_cooc,0))
	elif norm_method == 'hysteresis':
		c = np.prod(lin_cooc.shape)*lin_cooc/np.sum(np.sum(lin_cooc))
		c2 = c
		c2[c2>1] = 1
		c3 = np.prod(c2.shape)*c2/np.sum(np.sum(c2))
		lin_cooc = c3
	elif norm_method == 'binarize':
		lin_cooc = np.where(lin_cooc>0,1,0)
	else:
		print('No normalization method')
	print("test")

	lin_cooc = lin_cooc / np.amax(np.amax(lin_cooc));

	#Helper functions for losses
	col_diff_mat = None	
	for k in range(1,num_cast+1):
		addrows = num_cast - k
		mid_mat = np.column_stack([np.zeros([addrows,k-1]),np.ones([addrows,1])])
		diag_mat = np.diag(-1*np.ones(addrows))
		new_mat = np.column_stack([mid_mat,diag_mat])
		if col_diff_mat is None:
			col_diff_mat = new_mat
		else:
			col_diff_mat = np.vstack([col_diff_mat, new_mat])

	minsep_val =0.09

	#Se and ALl presence
	orig_presence = presence
	se_presence = np.zeros_like(presence)
	for k in range(num_cast):
		inds = np.argwhere(presence[k,:]==1)
		inds.sort()
		inds = inds.reshape([inds.shape[0],])
		if len(inds) == 1:
			se_presence[k,inds[0]] = 1
			continue
		se_presence[k,inds[0]:inds[-1]+1] = 1	
	
	logical_col_diff_mat = np.where(col_diff_mat!=0,1,0)
	new_se_mat = logical_col_diff_mat@se_presence
	diff_se_presence = 	(new_se_mat == 2)
	new_mat = logical_col_diff_mat@presence
	diff_presence = (new_mat == 2)
	
	all_presence = {}
	all_presence['orig'] = presence
	all_presence['se'] = se_presence
	all_presence['diff'] = diff_presence
	all_presence['diff_se'] = diff_se_presence

	return all_presence, init_indices, lin_cooc, col_diff_mat


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
	all_presence, init_indices, lin_cooc, col_diff_mat = get_cooccurence_info(block_times, cast_list, face_data)
	num_cast, n_scenes = init_indices.shape
	presence = all_presence['orig'] 
	se_presence = all_presence['se']
 
	np.save('./PyData/BBT_S1_ep1_presence.npy',presence)
	np.save('./PyData/BBT_S1_ep1_se_presence.npy',se_presence)
	
	