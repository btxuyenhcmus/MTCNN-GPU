import numpy as np
import torch

def conv_forward(input, weight, bias, stride, padding, dilation, groups):
    input = input.detach().numpy()
    weight = weight.detach().numpy()
    i_x, i_y, i_z, i_w = input.shape
    w_x, w_y, w_z, w_w = weight.shape
    result = np.random.rand(i_x, w_x, i_z + padding[0] * 2 - stride[0] * 2, i_w + padding[1] * 2 - stride[1] * 2).astype(np.float32)
    # TODO
    # implementation jit function
    return torch.from_numpy(result)