import numpy as np
import scipy.signal

def discount_cumsum(x, discount):
	#computes discounted cumulastive sum of vector x
	#line 45 in core of VPG 
	return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]

