from shot_boundary_detection import get_shots_arr_from_file
import numpy as np
import os
import cv2

def mean_rgb_hist_single_shot(shot_st_frame,shot_en_frame):
    num_frames = shot_en_frame-shot_st_frame+1
    n_bins = 6
    truncate = 1 
    mean_rgb_hist = np.zeros(n_bins**3)
    for frame_num in range(shot_st_frame,shot_en_frame+1):
        #Read each image frame and compute its RGB histogram
        filename = '../frames_generated/episode_'+str(frame_num).zfill(5)+'.jpg'
        if not os.path.exists(filename) or not os.path.isfile(filename):
            continue
        img = cv2.imread(filename)
        #Each pixel has rgb value - which determines which bin it goes into
        cl_hist = np.zeros([n_bins,n_bins,n_bins]) 
  	    
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                b = img[i,j,0]
                g = img[i,j,1]
                r = img[i,j,2]
                
                samp = 256/6
                b_num = int(b/samp)
                g_num = int(g/samp)
                r_num = int(r/samp)
                
                cl_hist[r_num,g_num,b_num] +=1

        normalized_cl_hist = cl_hist.reshape(-1) #Might have to change this
        normalized_cl_hist = normalized_cl_hist/np.sum(normalized_cl_hist)
        
        normalized_cl_hist[normalized_cl_hist>truncate]=truncate
        normalized_cl_hist = normalized_cl_hist/np.sum(normalized_cl_hist)
         
        mean_rgb_hist = mean_rgb_hist + normalized_cl_hist
   
    mean_rgb_hist = mean_rgb_hist/num_frames 
    
    return mean_rgb_hist

    
