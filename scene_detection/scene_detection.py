from shot_hist_representation import all_shots_representation
#from shot_threading import *
import argparse
import os
import numpy as np
from sklearn.metrics import pairwise_distances as ep_dists
import matplotlib.pyplot as plt
from kneed import KneeLocator

def sigmoid(x):
    return 1/(1+np.exp(-x))

def compute_dp_matrices(all_rep, shot_threading_sa):

    n_shots = all_rep.shape[0]
    n_scenes = int(np.floor(n_shots/5))
    n_layers = 100
    all_layers = np.array(range(n_layers+1)) #k = [1,.. Nl]
    alpha = 1 - 0.5*(all_layers**2/n_layers**2)#decay_factor

    print('No of shots', n_shots)
    print('No of scenes', n_scenes) 
    nr = n_scenes
    nc = n_shots
    nbins = 6**3 

    D_cube = np.zeros([nr+1,nc+1,n_layers+1])
    idx_cube = np.zeros([nr+1,nc+1,n_layers+1])
    
    #Create a hashmap for storing colorbased shot similarity scores
    hashmap = {}

    #Initialization of D_cube must be done properly.
    D_cube[:,1,1].fill(1) #Should all ones 
    
    #For Nsc = 1. D_cube must be filled accordingly. 
    i = 1
    for j in range(2,min(nc+1,n_layers+1)):
        k = j
        same_scene_score = 0
        
        #Color based similarity score
        this_shot = j-1
        prev_shots = np.array(range(j-k+2,j-1))
        prev_shots = prev_shots[prev_shots>=0] 
        if len(prev_shots) == 0:
            continue

        key = np.array2string(prev_shots)
        if key not in hashmap:
            hist_this = np.array([all_rep[this_shot]])
            hist_prevs = np.array([all_rep[sh] for sh in list(prev_shots)])
            shot_dist = np.mean(ep_dists(hist_this,hist_prevs))
            shot_score = sigmoid(shot_dist) #Can add -a,b here
            hashmap[key] = shot_score

        same_scene_score = hashmap[key]
        
        threading_factor = 0 
        #Compute threading factor from shot_threading_sa matrix
        if shot_threading_sa[this_shot] !=1 :
            threading_factor = np.sum(shot_threading_sa[prev_shots]==shot_threading_sa[this_shot])

        same_scene_score = alpha[k]*(same_scene_score + threading_factor)
        D_cube[i,j,k] = D_cube[i,j-1,k-1]+same_scene_score
        idx_cube[i,j,k] = np.ravel_multi_index([[i],[j-1],[k-1]],D_cube.shape)
        
    for i in range(2,nr+1):
        for j in range(2,nc):
            for k in range(1,min(j+1,n_layers+1)):
                #For each layer D[i][j][k] represents the score such that
                #previous k-1 shots assigned to same scene
                if k == 1 : #It's a scene boundary
                    max_score = float("-inf")
                    this_shot = j-1
                    max_loc = 1
                    for loc in range(1,n_layers+1):
                        #Color based shot similarity score.
                        prev_shots = np.array(range(j+2-loc,j-1))
                        prev_shots = prev_shots[prev_shots>0]
                        if len(prev_shots) == 0:
                            continue
                        key = np.array2string(prev_shots)
                        if key not in hashmap:
                            hist_this = np.array([all_rep[this_shot]])
                            hist_prevs = np.array([all_rep[sh] for sh in list(prev_shots)])
                            shot_dist = np.mean(ep_dists(hist_this,hist_prevs))
                            shot_score = sigmoid(shot_dist) #Can add -a,b here
                            hashmap[key] = shot_score
                        
                        score = 1 - hashmap[key]  
                        #Threading    
                        cap_threading = 0 #or 0
                        if shot_threading_sa[this_shot]!=1:
                            threading_factor = np.sum(shot_threading_sa[prev_shots]==shot_threading_sa[this_shot])
                            if threading_factor != 0:
                                score = max(0,score - 1)
                        
                        beta = 1-alpha[loc] #                  
                        score = beta*score + D_cube[i-1,j-1,loc]                        
                        if score > max_score:
                            max_score = score
                            max_loc = loc
                    D_cube[i,j,k] = max_score
                    idx_cube[i,j,k] = np.ravel_multi_index([[i-1],[j-1],[max_loc]],D_cube.shape)

                else: #k-1 shots are in the same scene as me. 
                    this_shot = j-1
                    prev_shots = np.array(range(j-k+2,j-1))
                    prev_shots = prev_shots[prev_shots>=0] 
                    if len(prev_shots) == 0:
                        continue
                    key = np.array2string(prev_shots)
                    if key not in hashmap:
                        hist_this = np.array([all_rep[this_shot]])
                        hist_prevs = np.array([all_rep[sh] for sh in list(prev_shots)])
                        shot_dist = np.mean(ep_dists(hist_this,hist_prevs))
                        shot_score = sigmoid(shot_dist) #Can add -a,b here
                        hashmap[key] = shot_score

                    same_scene_score = hashmap[key]
                    threading_factor = 0 
                    #Compute threading factor from shot_threading_sa matrix     
                    if shot_threading_sa[this_shot] !=1 :
                        threading_factor = np.sum(shot_threading_sa[prev_shots]==shot_threading_sa[this_shot])

                    same_scene_score = alpha[k]*(same_scene_score + threading_factor)
                    D_cube[i,j,k] = D_cube[i,j-1,k-1]+same_scene_score
                    idx_cube[i,j,k] = np.ravel_multi_index([[i],[j-1],[k-1]],D_cube.shape)
            print("No of shots",j)
        print("No of scenes,",i) 
 
    return D_cube,idx_cube

def find_elbow_point(vec):

    data = np.array([np.array(range(1,len(vec))),vec[1:]])
    print((data.T).shape)

    kn = KneeLocator(data.T[:,0],data.T[:,1], curve='concave', direction='increasing')
    elbow_index = int(kn.knee)

    return elbow_index

def backtrack_for_bounds(D_cube,I_cube,n_scenes):
     
    #Find the number of scenes if not specified
    if n_scenes == -1: # Method 1 : From Threshold
        threshold = 0.8
        vec = np.max(D_cube[:,-1,:],axis =1) 
        #Taking Direct max, gave N_scenes=29
        diffs = np.diff(vec) 
        n_scenes = np.argwhere(diffs<threshold)[0][0]+1
        print("N scenes", n_scenes)#17  
    elif n_scenes == -2:  #Method 2: Elbow Point
        vec = np.max(D_cube[:,-1,:],axis =1)
        n_scenes = find_elbow_point(vec) #vec is 1 based indexing
        print("N scenes", n_scenes)#7
    
    #Backtracking.
    _, nc,_ = D_cube.shape
    n_shots = nc-1 #366 shots
    #go to D_cube[n_scenes,n_shots,:], find max k.
    k = np.argmax(D_cube[n_scenes,n_shots,:])
    #Last boundary = new shot starts at. shot_number :0 based 
    #n_shots-k, n_shots-1
    end = n_shots-1
    path = []
    while n_scenes>0 and n_shots>0:
        if k==1:
            start = n_shots-1
            path.insert(0,[start,end, end-start+1])
            end = n_shots-2
        n_scenes,n_shots,k = np.unravel_index(I_cube[n_scenes,n_shots,k],D_cube.shape)
    
    path.insert([0,end, end-start+1])
    return np.array(path)
def get_scene_bounds(out_folder, input_video): 
    
    movie_name = input_video.split('/')[-1] 
    movie_name = movie_name.split('.')[-2]
    
    scene_file = os.path.join(out_folder,movie_name+'_1scene_bounds.npy')
    #Load from cache if present
    if os.path.exists(scene_file) and os.path.isfile(scene_file):
        scene_bounds = np.load(scene_file)
        print("Scene_bounds",scene_bounds.shape)
        return scene_bounds

    print('Computing Shots Representation ...') 
    all_rep = all_shots_representation(out_folder, input_video)
    print('Performing Shot Threading ...')
    #Change here
    shot_SA = np.load('../outputs/BBT_S1_ep1_shot_threading_sa.npy')[0]

    outfile_D = os.path.join(out_folder,movie_name+'_DTW_CUBE.npy')
    outfile_I = os.path.join(out_folder,movie_name+'_IDX_CUBE.npy')
    #Load from cache if present
    if os.path.exists(outfile_D) and os.path.exists(outfile_I):
        D_cube = np.load(outfile_D)
        I_cube = np.load(outfile_I)
        
    else:   
        D_cube,I_cube = compute_dp_matrices(all_rep,shot_SA)
        np.save(outfile_D,D_cube)
        np.save(outfile_I,I_cube)

    #Add backtracking stuff here.
    n_scenes = 6
    scene_bounds = backtrack_for_bounds(D_cube, I_cube, n_scenes)
 
parser = argparse.ArgumentParser(description="Scene Boundary Detection")

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
    
    print('Performing Scene Detection ...')
    scene_boundaries = get_scene_bounds(folder_path, movie_path)
    print('END') 
