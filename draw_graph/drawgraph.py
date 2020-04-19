import numpy as np
import scipy as sp
import os
import matplotlib.pyplot as plt

def mySigmoid(x,x1,x2):
	p = 6.9068
	a = 2*p/(x2-x1)
	b = p* (x2+x1)/(x2-x1)
	y = 1/(1+np.exp(-a*x+b))
	return y

def add_transition_segment(indices, ii,jj,presence, xcoord_forward_fillers):
	sigmoid_n_points = 20
	xc = xcoord_forward_fillers[jj,:]
	
	if presence[ii,jj] == presence[ii,jj+1]:
		xcoords = np.linspace(xc[0],xc[1],sigmoid_n_points).reshape(-1)	
		ycoords = mySigmoid(xcoords, xcoords[0], xcoords[-1])
		ycoords = (indices[ii,jj+1] - indices[ii,jj])*ycoords

	elif presence[ii,jj] and not presence[ii,jj+1]	:
		xcoords = np.concatenate(([xc[0]],np.linspace(xc[2],xc[1],sigmoid_n_points).reshape(-1)))
		ycoords = np.concatenate(([0],mySigmoid(xcoords[1:], xcoords[1], xcoords[-1]).reshape(-1)))
		ycoords = (indices[ii,jj+1] - indices[ii,jj])*ycoords

	elif presence[ii,jj+1]:		
		
		xcoords = np.concatenate((np.linspace(xc[0],xc[2],sigmoid_n_points).reshape(-1),[xc[1]]))
		ycoords = np.concatenate((mySigmoid(xcoords[:-1], xcoords[0], xcoords[-2]).reshape(-1),[1]))
		ycoords = (indices[ii,jj+1] - indices[ii,jj])*ycoords

	return xcoords, ycoords+indices[ii,jj]

def draw_graph(final_indices, block_times, presence, se_presence, cast_list):
	#change colors to random
	colors = ['r','g','b','c','y','m']
	num_cast, n_scenes = final_indices.shape
	fig = plt.figure()
	
	#Draw vertical lines
	for j in range(n_scenes):
		x1 = block_times[j,0]
		x2= block_times[j,0]
		y1 = 0
		y2 = num_cast
		lw = 0.5
		plt.plot([x1,x2],[y1,y2], color='black', linewidth=lw)
	x1 = block_times[n_scenes-1,1]
	plt.plot([x1,x1],[y1,y2], color='black', linewidth=lw)
	
	off = 0.25
	index_duration = block_times[:,1]-block_times[:,0]
	xcoord_startend = np.column_stack([off * index_duration+block_times[:,0], (1 - off) * index_duration +block_times[:, 0]])
	xcoord_forward_fillers = np.column_stack([(1-off) * index_duration[0:-1] + block_times[0:-1, 0],
                            off * index_duration[1:]   + block_times[0:-1, 1], 
                            block_times[0:-1, 1]])
	#xcoord_forward_fillers.		
	for i in range(num_cast):
		n = 0
		for j in range(n_scenes):
			#Se presence
			if se_presence[i,j] == 0:
				continue
			#change this x1,x2
			x1 = xcoord_startend[j,0]
			x2 = xcoord_startend[j,1]	
			if n == 0:
				x1 = block_times[j,0]
			else:
				if j == n_scenes-1 or presence[i,j+1] == 0:
					x2 = block_times[j,1]
				if presence[i,j-1] == 0 and presence[i,j] == 1:
					x1 = block_times[j,0]
			if j==n_scenes-1 or se_presence[i,j+1]==0:
				n=2
				print("last appearance of ",cast_list[i],j+1)
			if n == 2:
				x2 = block_times[j,1]
			y1 = final_indices[i,j]
			y2 = y1
			#Add marker based on presence
			m = 'dotted'
			lw = 0.5
			if presence[i,j] == 1:
				m = 'solid'
				lw  = 2
			plt.plot([x1,x2],[y1,y1], color=colors[i], linestyle=m, linewidth=lw, label=cast_list[i] if n==0 else"")
			if n == 0:
				plt.scatter(x1,y1, color=colors[i], marker='o')
			if n == 2:
				plt.scatter(x2,y1, color=colors[i], marker='>')
			if j < n_scenes -1:
				#Add transition segment
				m = 'dotted'
				lw = 0.5
				if presence[i,j] and  presence[i,j+1] == 1:
					m = 'solid'
					lw  = 2
				xcoords, ycoords = add_transition_segment(final_indices,i,j,presence,xcoord_forward_fillers)
				plt.plot(xcoords,ycoords, color=colors[i], linestyle=m, linewidth=lw)
			
			n = 1
	plt.legend()
	plt.show()	
	


if __name__ == '__main__':
	folder = './PyData/'
	movie_name = 'BBT_S1_ep1'
	#Load xct
	xct = np.load(os.path.join(folder,movie_name+'_final_coords.npy'))
	#Load blocktimes
	block_times = np.load(os.path.join(folder,movie_name+'_block_times.npy'))
	#Load presence?
	presence = np.load(os.path.join(folder,movie_name+'_presence.npy'))
	#Load se_presence
	se_presence = np.load(os.path.join(folder,movie_name+'_se_presence.npy'))
	DataFold = './PyData'
	cast_file = open(os.path.join(DataFold,movie_name+'_castlist.dat'),'r')	
	cast_list = cast_file.read().split(' \n')
	cast_file.close()
	cast_list = cast_list[:-1] #had empty line at last

	draw_graph(xct,block_times,presence, se_presence, cast_list)	
