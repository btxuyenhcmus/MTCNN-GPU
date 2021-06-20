import torch
import numpy

def Convolution2D(input,weight,bias,strike,padding,dilation,groups):
    assert(dilation == 1)
    assert(groups == 1)

    n_input=input.data.cpu().numpy()
    n_weight=weight.data.cpu().numpy()
    n_bias=bias.data.cpu().numpy()

    N, Kc, H, W = n_input.shape
    C, Kc, Kh, Kw = n_weight.shape

    np_padding = np.pad(n_input, pad_width=((0,),(0,),(padding,),(padding,)), mode='constant', constant_values=0)

    N, Kc, H, W = n_input.shape
    C, Kc, Kh, Kw = n_weight.shape
    W_out = int(np.floor((W + 2*padding - dilation*(Kw-1) - 1) / stride + 1))
    H_out = int(np.floor((H + 2*padding - dilation*(Kh-1) - 1) / stride + 1))

    res = np.zeros((N, C, H_out, W_out), dtype=np.float32)