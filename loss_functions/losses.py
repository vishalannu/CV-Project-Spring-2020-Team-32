import numpy as np
import scipy as sp
from custom_losses import zeroed_inverse_loss, soft_hinge_loss

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
	
	#Proximity subtract loss
	tilde_lin_cooc = np.where(lin_cooc==0,1,0)
	m3 = np.multiply(diff_indices**2,tilde_lin_cooc)	
	Lp2 = np.sum(np.sum(m3))

	#Minimum separation loss
	m4 = np.multiply(diff_se_presence, zeroed_inverse_loss(diff_indices**2+1e-6,minsep_val))

	Ls = np.sum(np.sum(m4))

	#Crossing loss
	m5 = np.multiply(diff_indices[:,:-1],diff_indices[:,1:])
	m6 = np.multiply(diff_se_presence[:,:-1],diff_se_presence[:,1:])
	valid_crossing_scores = np.multiply(soft_hinge_loss(m5,0,1),m6)
	Lc = np.sum(np.sum(valid_crossing_scores))
	

	wcross = 1
	wsep = 1
	wprox = 1
	wppush = 0.1 #Note this needs to be subtracted
	wstraight = 1


	norm_crossing_loss = Lc /(nr*(nc-1))
	norm_minsep_loss = Ls /(nd*nc)
	norm_prox_loss = Lp1/(nd*nc)
	norm_proxpush_loss = Lp2/(nd*nc)
	norm_straight_loss = Ll/(nd*(nc-1))
	
	#print("Normalized losses")
	#print("NLc",norm_crossing_loss, end="|")
	#print("NLs",norm_minsep_loss,end="|")
	#print("NLp1",norm_prox_loss,end="|")
	#print("NLp2",norm_proxpush_loss,end="|")
	#print("NLl",norm_straight_loss)
	
	obj = norm_crossing_loss*wcross + norm_minsep_loss*wsep + \
			norm_prox_loss*wprox  - norm_proxpush_loss*wppush + \
			norm_straight_loss*wstraight

	return obj
