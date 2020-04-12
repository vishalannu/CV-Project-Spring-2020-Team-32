import numpy as np
import scipy as sp
from custom_losses import zeroed_inverse_loss

def loss_function(indices, lin_cooc, allpresence,col_diff_mat):

	presence = allpresence['orig']
	se_presence = allpresence['se']
	diff_presence = allpresence['diff']
	diff_se_presence = allpresence['diff_se']
	
	xct = indices.reshape(presence.shape)
	#xct vector is [Nc,Nscenes] vector
	#output a scalar
	
	num_cast, n_scenes = presence.shape
	diff_indices = col_diff_mat@xct;
	minsep_val  = 0.09
	
	#What are nr,nc and nd
	nr = num_cast
	nc = n_scenes
	nd = nr*(nr-1)/2

	#Straight Line Loss
	Ll = 0
	for i in range(nr):
		med = np.multiply(se_presence[i,:],xct[i,:]).reshape([n_scenes,])
		this_line = med[np.nonzero(med)]#All non zero values
		if len(this_line) == 1:
			continue
		mean_line = np.zeros(len(this_line))
		for k in range(len(this_line)):
			mean_line[k] = (np.sum(this_line) - this_line[k])/(len(this_line) -1)
		Ll = Ll + np.sum((this_line-mean_line)**2)/nr	
		
	#Proxmity add loss
	m1 = np.multiply(diff_presence, diff_indices**2)
	m2 = np.multiply(m1,lin_cooc)
	Lp1 = np.sum(np.sum(m2))
	
	wprox = 1
	wstraight = 1


	norm_prox_loss = Lp1/(nd*nc)
	norm_straight_loss = Ll/(nd*(nc-1))
	
	#print("Normalized losses")
	#print("NLp1",norm_prox_loss,end="|")
	#print("NLl",norm_straight_loss)
	
	obj = norm_prox_loss*wprox + norm_straight_loss*wstraight

	return obj
