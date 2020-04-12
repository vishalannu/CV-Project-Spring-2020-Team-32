import numpy as np

def zeroed_inverse_loss(inp, offset=1):
	'''In Math form
		f(x) = 0, x>= offset	
			 = 1/x* (sqrt(1+(x-offset)^2)-1) , otherwise
	'''
	f = np.zeros(inp.shape)
	common = np.sqrt(1+(inp-offset)**2)
	nonzeroidx = inp < offset #wherever inp < offset
	f[nonzeroidx] = np.divide( common[nonzeroidx]-1, inp[nonzeroidx])	
	return f

 
