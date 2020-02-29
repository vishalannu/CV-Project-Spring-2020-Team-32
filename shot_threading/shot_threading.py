import cv2
import sys
import os.path
import numpy as np
import networkx  as nx

def are_images_similar(filename1, filename2):
    img1 = cv2.imread(filename1)          # queryImage
    img2 = cv2.imread(filename2)          # trainImage

    # Initiate SIFT detector
    sift = cv2.SIFT()

    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)

    # BFMatcher with default params
    bf = cv2.BFMatcher()
    matches = bf.match(des1,des2)

	# if len(matches) > threshold ; True
	threshold = 20
	if len(matches) > threshold:
		return True
	return False

def compute_shot_similarity_graph(shots_arr):

	#Add cache load here.
	lookahead = 24
	G = nx.Graph()
	#Add nodes in the graph - no of shots.
	G.add_nodes_from(list(range(n_shots)))
	
	#For each shot k, k+1,k+r
	n_shots = shots_arr.shape[0]
	for k in range(n_shots):
		for r in range(1,lookahead+1):
			last_of_first = shots_arr[k,1]
			first_of_second = shots_arr[k+r,0]

			#Those are frame numbers, compute filenames from this.
			f1 = 'bbt_s01e01_'+str(last_of_first).zfill(6)+'.jpg'
			f1 = os.path.join('../data',f1)
			
			f2 = 'bbt_s01e01_'+str(first_of_second).zfill(6)+'.jpg'
			f2 = os.path.join('../data',f2)
			
			decision = are_images_similar(f1,f2)
			if decision == True:
				G.add_edge(k,k+r)
				G.add_edge(k+r,k)
	
	return G

if __name__ == '___main__':
	#Get shots_arr from shot_boundary_detection
	compute_shot_similarity_graph(shots_arr)
