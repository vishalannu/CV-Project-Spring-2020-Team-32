import numpy as np
import scipy as sp
from losses import loss_function

def storygraph_optimization(castlist, indices, all_presence, lin_cooc, col_diff_mat, block_times):


	num_cast = len(castlist)
	n = len(indices)
	n_scenes = int(n/num_cast)
	ind_bounds = tuple([(1,num_cast)]*n)
		
	optimoptions = {'maxiter':1000,'disp': True}
	for i in range(0):
		if i == 4:
			optimoptions['maxiter'] = 300
		else:
			optimoptions['maxiter']=50

		res = sp.optimize.minimize(loss_function,indices,(lin_cooc, all_presence, col_diff_mat),method='TNC',bounds= ind_bounds, tol=1e-6, options = optimoptions)
		indices = res.x

		indices1 =  indices.copy()
		indices1 = indices1.reshape([num_cast, n_scenes])
		pSGindices = perturb_sg_indices(indices1, lin_cooc, all_presence, col_diff_mat)
		indices = pSGindices.reshape([n,])
	
	res = sp.optimize.minimize(loss_function,indices,(lin_cooc, all_presence, col_diff_mat),method='TNC',bounds= ind_bounds, tol=1e-6, options = optimoptions)
	indices = res.x
	
	return res

def perturb_sg_indices(indices,lin_cooc, all_presence, col_diff_mat):
	n = np.prod(indices.shape)
	indices1 = indices.reshape([n,])
	init_obj = loss_function(indices1, lin_cooc, all_presence, col_diff_mat)

	bestobj = init_obj
	SG = [indices, lin_cooc, all_presence, col_diff_mat]
	while True:
		candidates = search_crossing_candidates(SG)
		SG, newobj = run_swapping_pass(SG, candidates, bestobj)
		if abs(bestobj - newobj) <1e-10:
			break
		else:
			bestobj = newobj
	
	pSGindices, _, _, _ = SG

	return pSGindices

def search_crossing_candidates(SG):
	indices, lin_cooc, all_presence, col_diff_mat = SG

	se_presence = all_presence['se']
	diff_indices = col_diff_mat@indices
	crossings_at = np.multiply(diff_indices[:,:-1],diff_indices[:,1:]) < 0

	m1 = se_presence@se_presence.T
	i,j = np.nonzero(m1); #change the find
	candidates = np.column_stack([i,j,np.zeros([len(i),1])])

	for t in range(crossings_at.shape[1]):
		m2 = sp.spatial.distance.squareform(crossings_at[:,t])	
		i,j = np.nonzero(np.triu(m2)) #change this find
		m3 = np.column_stack([i,j,(t+1)*np.ones([len(i),1])])
		candidates = np.vstack([candidates, m3])
	
	return candidates

def run_swapping_pass(SG, candidates, bestobj):
	indices, lin_cooc, all_presence, col_diff_mat = SG
	n = np.prod(indices.shape)
	for k in range(candidates.shape[0]):
		pSGindices = indices
		i, j, t = candidates[k,:3].astype(np.int)
		#Swap i, j
		temp = 	indices[i,t]
		indices[i,t] = indices[j,t]
		indices[j,t] = temp
		
		indices1 = indices.reshape([n,])
		currobj = loss_function(indices1, lin_cooc, all_presence , col_diff_mat)
		if currobj >=bestobj:
			indices = pSGindices		
		else:
			bestobj = currobj

	SG = [indices, lin_cooc, all_presence, col_diff_mat]
	return SG, bestobj
