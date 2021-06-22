import numpy as np
import torch
from numba import jit, prange

def conv_forward_type1(input, weight, bias, stride, padding, dilation, groups):
    np_input = input.data.cpu().numpy()
    np_input_pad = np.pad(input, pad_width=((0,), (0,), (padding[0],), (padding[1],)), mode='constant', constant_values=0)
    np_weight = weight.data.cpu().numpy()
    np_bias = bias.data.cpu().numpy()
    
    # Dimensions.
    N, Kc, H, W = input.shape
    C, Kc, Kh, Kw = weight.shape
    W_out = int(np.floor((W + 2*padding[0] - dilation[0]*(Kw-1) - 1) / stride[0] + 1))
    H_out = int(np.floor((H + 2*padding[1] - dilation[1]*(Kh-1) - 1) / stride[1] + 1))
    result = np.zeros((N, C, H_out, W_out), dtype=np.float32)
    _conv_forward_type1(np_input, np_input_pad,np_weight, np_bias, stride, padding, dilation, groups, result)
    return torch.from_numpy(result).type_as(input)

def prelu_type1(input, weight):
	np_input = input.data.cpu().numpy()
	np_weight = weight.data.cpu().numpy()
	result = np.zeros(np_input.shape, dtype=np.float32)
	_prelu_type1(np_input, np_weight, result)
	return torch.from_numpy(result).type_as(input)

@jit
def _conv_forward_type1(input, input_pad, weight, bias, stride, padding, dilation, groups, result):
    N, Kc, H, W = input.shape
    C, Kc, Kh, Kw = weight.shape
    W_out = int(np.floor((W + 2*padding[0] - dilation[0]*(Kw-1) - 1) / stride[0] + 1))
    H_out = int(np.floor((H + 2*padding[1] - dilation[1]*(Kh-1) - 1) / stride[1] + 1))
    for n in range(N):
      for to in range(C):
        for y in range(H_out):
          for x in range(W_out):
            for ti in range(Kc):
              for ky in range(Kh):
                for kx in range(Kw):
                  result[n,to,y,x] += weight[to,ti,ky,kx] * input_pad[n,ti,(y*stride[1])+ky,(x*stride[0])+kx]  
            result[n,to,y,x] += bias[to]

@jit
def _prelu_type1(input, weight, result):
	if len(input.shape) == 2:
		result = input
	N, Kc, H, W = input.shape
	for n in range(N):
		for to in range(Kc):
			for y in range(H):
				for x in range(W):
					if input[n, to, y, x] >= 0:
						result[n, to, y, x] = input[n, to, y, x]
					else:
						result[n, to, y, x] = input[n, to, y, x] * weight[to]