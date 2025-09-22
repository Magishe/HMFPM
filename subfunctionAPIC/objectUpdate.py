import torch
import numpy as np
import matplotlib.pyplot as plt

def objectUpdate(O, P, dpsi,  cen,  alpha, gamma, ftMatching_Mask,OP_noise_th, P_abs2, P_mask, updating_full_mask):
    """
    Updates the estimate of O and P.
    """

    # Size of P
    Np = dpsi.shape[0]
    r = dpsi.shape[2]  # Number of slices along third dimension
    sumP = torch.zeros_like(O, device = 'cuda')
    O_New = torch.zeros_like(O, device = 'cuda')
    mask = torch.zeros_like(O, dtype=torch.bool, device = 'cuda')
    submask = P_abs2 > OP_noise_th
    n1 = cen -np.int32(Np / 2)
    n2 = n1 + Np
    dO0 = P.unsqueeze(2).conj() * dpsi

    for m in range(r):
        mask[n1[0,m]:n2[0,m], n1[1,m]:n2[1,m]] += submask
        O_New[n1[0,m]:n2[0,m], n1[1,m]:n2[1,m]] += dO0[:, :, m]*submask
        sumP[n1[0,m]:n2[0,m], n1[1,m]:n2[1,m]] += P_abs2*submask

    # Update spectrum after processing every image
    updating_mask = mask & ~ftMatching_Mask
    O_New = (1 / P_mask) * O_New / (sumP+gamma)
    O[updating_mask] = alpha * O_New[updating_mask]
    updating_full_mask = mask | updating_full_mask
    return O, updating_full_mask
