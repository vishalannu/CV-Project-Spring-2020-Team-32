#!/usr/bin/env python
# coding: utf-8

# In[14]:


import cv2
import glob

# In[ ]:

def dfd_func(image1,image2):
	img1 = cv2.imread('image1')
	img2 = cv2.imread('image2')

	


def shot_boundary(images_input):
    
	frames_array = []
    
	ans = glob.glob(images_input+'/*.jpg')
	ans.sort(key= lambda x:int(x[-9:-4]))
	print(ans[0])

	num_of_frames = len(ans)

	DFD_score_arr = []
	score = 0
	
	num_of_frames = 1

	for i in range(1,num_of_frames):
		score = dfd_func(ans[i],ans[i-1])
		DFD_score_arr.append(score)

    


# In[28]:


def generating_frames(input_video,out_folder,img_name):
	inp = cv2.VideoCapture(input_video)
	while True :
		j = 1
		ret, frame = inp.read()
		if not ret:
			break
			cv2.imwrite(out_folder+img_name +str(j).zfill(5)+'.jpg',frame)
			j=j+1
#         num+=1
	inp.release()
	cv2.destroyAllWindows()


# In[32]:


def main():
	input_video = './input/BBT_S1_ep1.avi'
	out_folder = './frames_generated/'
	img_name = 'episode_'
	generating_frames(input_video,'./frames_generated/',img_name)
	generating_frames(input_video,out_folder,img_name)
	shot_boundary('./frames_generated/')

# In[35]:


if __name__ == '__main__':
	main()


# In[ ]:




