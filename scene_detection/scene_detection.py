from shot_hist_representation import *
#from shot_threading import *
from sklearn.metrics import pairwise_distances as ep_dists
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

def get_scene_bounds(all_rep, shot_thread_sa, n_scenes=None):
    
    outfile_DTW = '../outputs/BBT_S1_ep1_DTW_CUBE.npy'
    outfile_IDX = '../outputs/BBT_S1_ep1_IDX_CUBE.npy'
    scene_file = '../outputs/BBT_S1_ep1_scene_bounds.npy'
    if os.path.exists(outfile_DTW) and os.path.exists(outfile_IDX):
        #saved representation exists read that and return 
        dtw_cube = np.load(outfile_DTW)
        idx_cube = np.load(outfile_IDX)
        
    else:    
        dtw_cube,idx_cube = compute_dp_matrices(all_rep,shot_thread_sa)
        np.save(outfile_DTW,dtw_cube)
        np.save(outfile_IDX,idx_cube)

    if os.path.exists(scene_file) and os.path.isfile(scene_file):
        scene_bounds = np.load(scene_file)
        return dtw_cube,idx_cube, scene_bounds

    #Find the number of scenes if not specified
    if n_scenes == None: #Two methods to detect the scenes
        #Method 1 : From Threshold
        threshold = 0.8
        vec = np.max(dtw_cube[:,-1,:],axis =1)
        diffs = np.diff(vec)
        n_scenes = np.argwhere(diffs<threshold)[0][0]+1    
    n_shots = len(all_rep)
    #Do backtracking to find scene boundaries
    ii = n_scenes+1
    jj = n_shots +1
    kk = np.argmax(dtw_cube[ii-1,jj-1,:])+1
    path = [[ii-1,jj-1,kk,dtw_cube[ii-1,jj-1,kk-1]]]
    while ii>1 and jj>1:
        loc = idx_cube[ii-1,jj-1,kk-1] 
        if not loc:
            break
        ii,jj,kk = np.unravel_index(loc,dtw_cube.shape,'F')
        path.insert(0,[ii-1,jj-1,kk,dtw_cube[ii-1,jj-1,kk]])
    for k in range(len(path)):
        if len(scene_bounds) != path[k][1]:
            scene_bounds = [scene_bounds,k]
    scene_bounds = np.array(scene_bounds)
    np.save(scene_file,scene_bounds)
    return dtw_cube,idx_cube, scene_bounds
 
if __name__ == '__main__' :
    shots_arr = get_shots_arr_from_file()
    all_rep = all_shots_representation(shots_arr)

    #sim = sift_shot_similarity(shots_arr)
    #_, shot_threading_sa = similarity_to_threads(sim)
    shot_threading_sa = np.load('../outputs/BBT_S1_ep1_shot_threading_sa.npy')
    scene_boundaries = get_scene_bounds(all_rep,shot_threading_sa[0]) 
