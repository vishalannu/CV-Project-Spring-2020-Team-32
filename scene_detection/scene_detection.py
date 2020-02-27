from shot_hist_representation import *
#from shot_threading import *
def sigmoid(x):
    return 1/(1+np.exp(-x))

def compute_dp_matrices(all_rep, shot_threading_sa):
    n_shots = len(all_rep)
    n_scenes = int(np.floor(n_shots/5))
    n_layers = 100
    all_layers = np.array(range(n_layers+1))
    alpha = 1 - 0.5*(all_layers**2/n_layers**2)#decay_factor
    
    nr = n_scenes+1
    nc = n_shots+1
    nbins = 6**3 

    dtw_cube = np.zeros([nr+1,nc+1,n_layers+1])
    idx_cube = np.zeros([nr+1,nc+1,n_layers+1])

    #Create a hashmap for storing colorbased shot similarity scores
    hashmap = {}
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
                        prev_shots = np.array(range(j-1-loc,j-1))
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
                        cap_threading = 0 #or 0
                        if shot_threading_sa[this_shot]!=1:
                            threading_factor = np.sum(shot_threading_sa[prev_shots]==shot_threading_sa[this_shot])
                            if threading_factor != 0:
                                score = max(0,score - 1)
                        beta = alpha[1-loc] #Using flipped alpha instead of 1- alpha                   
                        score = beta*score + dtw_cube[i-1,j-1,loc]                        
                        if score > max_score:
                            max_score = score
                            max_loc = loc
                        
                    dtw_cube[i,j,k] = max_score
                    idx_cube[i,j,k] = np.ravel_multi_index([[i-1],[j-1],[max_loc]],dtw_cube.shape)

                else: #k-1 shots are in the same scene as me. 
                    this_shot = j-1
                    prev_shots = np.array(range(j-k+1,j-1))
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

                    same_scene_score = hashmap[key]
                    threading_factor = 0 
                    #Compute threading factor from shot_threading_sa matrix
                    if shot_threading_sa[this_shot] !=1 :
                        threading_factor = np.sum(shot_threading_sa[prev_shots]==shot_threading_sa[this_shot])
                    
                    same_scene_score = alpha[k]*(same_scene_score + threading_factor)
                    dtw_cube[i,j,k] = dtw_cube[i,j-1,k-1]+same_scene_score
                    idx_cube[i,j,k] = np.ravel_multi_index([[i],[j-1],[k-1]],dtw_cube.shape)
            print("No of shots",j)
        print("No of scenes,",i) 
 
    return dtw_cube,idx_cube

def get_scene_bounds(all_rep,shot_SA ): 
	D_cube, I_cube = compute_dp_matrices()
 
if __name__ == '__main__' :
    shots_arr = get_shots_arr_from_file()
    all_rep = all_shots_representation(shots_arr)

    #sim = sift_shot_similarity(shots_arr)
    #_, shot_threading_sa = similarity_to_threads(sim)
    shot_threading_sa = np.load('../outputs/BBT_S1_ep1_shot_threading_sa.npy')
    scene_boundaries = get_scene_bounds(all_rep,shot_threading_sa[0]) 
