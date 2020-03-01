from shot_hist_representation import *
#from shot_threading import *

def sigmoid(x):
    return 1/(1+np.exp(-x))

def compute_dp_matrices(all_rep, shot_threading_sa):

    n_shots = all_rep.shape[0]
    n_scenes = int(np.floor(n_shots/5))
    n_layers = 100
    all_layers = np.array(range(n_layers+1)) #k = [1,.. Nl]
    alpha = 1 - 0.5*(all_layers**2/n_layers**2)#decay_factor
 
    nr = n_scenes
    nc = n_shots
    nbins = 6**3 

    D_cube = np.zeros([nr+1,nc+1,n_layers+1])
    idx_cube = np.zeros([nr+1,nc+1,n_layers+1])
    
    #Create a hashmap for storing colorbased shot similarity scores
    hashmap = {}

    #Initialization of dtw_cube must be done properly.
    D[:,1,1].fill(1) #Should all ones 
    
    #For Nsc = 1. dtw_cube must be filled accordingly. 
    for j in range(2,min(nc+1,nlayers+1)):
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

def get_scene_bounds(all_rep,shot_SA ): 
    D_cube, I_cube = compute_dp_matrices(all_rep, shot_SA)
 
if __name__ == '__main__' :
    shots_arr = get_shots_arr_from_file()
    all_rep = all_shots_representation(shots_arr)

    #sim = sift_shot_similarity(shots_arr)
    #_, shot_threading_sa = similarity_to_threads(sim)
    shot_threading_sa = np.load('../outputs/BBT_S1_ep1_shot_threading_sa.npy')
    scene_boundaries = get_scene_bounds(all_rep,shot_threading_sa[0]) 
