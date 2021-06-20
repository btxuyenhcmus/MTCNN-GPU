import torch
import numpy

def Convolution2D(input,weight,bias,strike,padding,dilation,groups):
    assert(dilation == 1)
    assert(groups == 1)

    