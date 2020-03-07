import cv2
import sys
import os.path
import numpy as np
import networkx  as nx
from shot_boundary_detection import shots_arr_from_DFD
import matplotlib.pyplot as plt
from itertools import product
import argparse

def are_images_similar(filename1, filename2):
	img1 = cv2.imread(filename1)		  # queryImage
	img2 = cv2.imread(filename2)		  # trainImage

	# Initiate SIFT detector
	sift = cv2.ORB_create()

	# find the keypoints and descriptors with SIFT
	kp1, des1 = sift.detectAndCompute(img1,None)
	kp2, des2 = sift.detectAndCompute(img2,None)

	if des1 is None or des2 is None:
		#print("error",filename1,filename2)
		return False, 0
	# BFMatcher with default params
	bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)	
	# Match descriptors.
	matches = bf.match(des1,des2)
	
	# Sort them in the order of their distance.
	matches = sorted(matches, key = lambda x:x.distance)

	n_matches = 0
	dist_thresh = 51
	for m in matches:
		if m.distance < dist_thresh:
			n_matches = n_matches+1
		else:
			break
#	print([m.distance for m in matches])
#	print(n_matches)
	# Draw first 10 matches.
	#img3 = cv2.drawMatches(img1,kp1,img2,kp2,matches[:10],None, flags=2)	

	#plt.imshow(img3)
	#plt.show()	
	
	# if len(matches) > threshold ; True
	threshold = 30

	if n_matches > threshold:
		return True, n_matches
	return False, n_matches

def compute_shot_similarity_graph(shots_arr, out_folder, input_video):

	movie_name = input_video.split('/')[-1]	
	movie_name = movie_name.split('.')[-2]
	sim_path = os.path.join(out_folder,movie_name+'_sim_graph.npy')
	if os.path.exists(sim_path) and os.path.isfile(sim_path):
		adj_mat = np.load(sim_path)
		G = nx.from_numpy_matrix(adj_mat)
		return G
	lookahead = 24
	G = nx.Graph()
	
	#Add nodes in the graph - no of shots.
	n_shots = shots_arr.shape[0]
	for i in range(n_shots):
		G.add_node(i)

	print("Nshots",n_shots)
	#G.add_edge(1,2)	
#	print(nx.adjacency_matrix(G))
	#For each shot k, k+1,k+r
	edges = 0
	for k in range(n_shots):
		edges = 0
		for r in range(k+1,min(k+lookahead+1,n_shots)):
			
			last_of_first = shots_arr[k,1]
			first_of_second = shots_arr[r,0]

			#Those are frame numbers, compute filenames from this.
			f1 = movie_name+'_'+str(last_of_first).zfill(6)+'.jpg'
			f1 = os.path.join(out_folder, 'frames/'+f1)
			
			f2 = movie_name+'_'+str(first_of_second).zfill(6)+'.jpg'
			f2 = os.path.join(out_folder, 'frames/'+f2)

			#print("f1,f2",f1,f2)			
			decision, num_matches = are_images_similar(f1,f2)
			if decision == True:
				G.add_edge(k,r)
				G.add_edge(r,k)
				edges = edges+1
		print("edges so far", edges)
		print("shots k",k)
	G1 = nx.adjacency_matrix(G, nodelist=range(G.number_of_nodes()))
	np.save(sim_path,np.array(G1.todense()))
	return G

def similarity_to_threads(Sim_Graph, out_folder, input_video):
	
	#Load from cache if present
	movie_name = input_video.split('/')[-1]	
	movie_name = movie_name.split('.')[-2]
	shots_file = os.path.join(out_folder,movie_name+'_shot_threading_sa.npy')
	if os.path.exists(shots_file) and os.path.isfile(shots_file):
		shots_assigned = np.load(shots_file)
		return shots_assigned

	#Find maximal cliques in SimGraph
	#if cliques overlap , apply transitivity.
	#Repeat this for max_times = 5
	max_times = 5
	cliqs =list(nx.find_cliques(G))
	adj_mat = nx.adjacency_matrix(G,nodelist=range(G.number_of_nodes()))
	adj_mat = adj_mat.todense()

	n_rows = adj_mat.shape[0]
	for i in range(max_times):
		#if cliques overlap. 
		flattened = np.array(sum(cliqs,[]))
		l2 = len(flattened)
		if l2 == n_rows :#cliques do not overlap
			break

		#Apply transitivity
		for k in range(n_rows):
			non_zero = np.nonzero(adj_mat[k,:])[1]# 1xN
			#indices of non_zero
			if len(non_zero)<1:
				continue
			pairs = product(non_zero,non_zero) 
			G.add_edges_from(pairs)
			
		#find cliqs again
		cliqs =list(nx.find_cliques(G))

	ele_in_cliq_index = [0]*n_rows
	for i in range(len(cliqs)):
		for j in cliqs[i]:
			ele_in_cliq_index[j] = i+1

	shots_assigned = np.array(ele_in_cliq_index)
	np.save(shots_file, shots_assigned)		
	return shots_assigned

def compute_shot_threads(out_folder, input_video):
	print("Getting the shot boundaries ...")
	shots_arr = shots_arr_from_DFD(out_folder,input_video)
	print("Computing Shot Similarity Graph ...")
	G =	compute_shot_similarity_graph(shots_arr, out_folder, input_video)
	print("Applying transitivity and finding cliques ...")
	shots_assigned = similarity_to_threads(G, out_folder, input_video)
	return shots_assigned

parser = argparse.ArgumentParser(description="Shot Threading")

parser.add_argument('--inp_file', type=str, help='Path of Input video file', required=True)
parser.add_argument('--out_folder', type=str, help='NAME of folder to store the output files',required=True)

if __name__ == '__main__':

	args = parser.parse_args()
	movie_path = args.inp_file
	folder_path = os.path.join(os.getcwd(),args.out_folder)

	print(folder_path)
	assert os.path.exists(movie_path), 'No file exists at '+ movie_path+ '.'

	if not os.path.exists(folder_path):
		os.mkdir(folder_path)
	assert os.path.isdir(folder_path),folder_path+ ' already exists and is not a directory. Please specify a different name.'

	print("Performing shot threading..")
	threads_shots_assigned = compute_shot_threads(folder_path, movie_path)
