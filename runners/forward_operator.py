import torch


class H_functions:
    """
    A class replacing the SVD of a matrix H, perhaps efficiently.
    All input vectors are of shape (Batch, ...).
    All output vectors are of shape (Batch, DataDimension).
    """
    def H(self, vec):
        """
        Multiplies the input vector by H
        """
        return self.H(vec)


# a memory inefficient implementation for any general degradation H
class GeneralH(H_functions):
    def mat_by_vec(self, M, v):
        vshape = v.shape[1]
        if len(v.shape) > 2: vshape = vshape * v.shape[2]
        if len(v.shape) > 3: vshape = vshape * v.shape[3]
        return torch.matmul(M, v.view(v.shape[0], vshape,
                                      1)).view(v.shape[0], M.shape[0])

    def H(self, vec):
        return self.mat_by_vec(self._H, vec.clone())


# Walsh-Hadamard Compressive Sensing
class GaussianChannel(H_functions):
    def __init__(self, channels, img_dim, ratio, perm, device):
        self.channels = channels
        self.img_dim = img_dim
        self.ratio = ratio
        self.perm = perm
        self._singulars = torch.ones(channels * img_dim ** 2 // ratio, device=device)

    def V(self, vec):
        temp = torch.zeros(vec.shape[0], self.channels, self.img_dim ** 2, device=vec.device)
        temp[:, :, self.perm] = vec.clone().reshape(vec.shape[0], -1, self.channels).permute(0, 2, 1)
        return self.fwht(temp).reshape(vec.shape[0], -1)



