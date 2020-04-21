from co_occurence_matrix import * do


linear = np.prod(init_indices.shape)
indices = init_indices.reshape([linear,])
LOSS = loss_function(indices, lin_cooc, all_presence, col_diff_mat)
print(LOSS)	

	############################################
res = storygraph_optimization(cast_list, indices, all_presence, lin_cooc, col_diff_mat, block_times )
final_indices = res.x
final_indices = final_indices.reshape([num_cast, n_scenes])
	
np.save('./PyData/BBT_S1_ep1_final_coords.npy',final_indices)
	
print("Final indices are:")
print(final_indices)
print("Final Loss is:")
FLOSS = loss_function(final_indices, lin_cooc, all_presence, col_diff_mat)
print(FLOSS)	

	############################
draw_graph(final_indices, block_times, presence, se_presence, cast_list)
