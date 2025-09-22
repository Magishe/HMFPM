import torch
import numpy as np
from subfunctionAPIC.calBoundary import calBoundary
def spectraStitching(ftRecons,maskRecons,recFTframe,CTF_abe_cuda,mask2use,k_illu_BF,useAbeCorrection):

    """
    Perform spectral stitching with aberration correction.
    """

    CTF_abe_abs_cuda = torch.abs(CTF_abe_cuda)
    xcR = ftRecons.shape[0] // 2 + 1
    ycR = ftRecons.shape[1] // 2 + 1

    [xsize, ysize] = CTF_abe_cuda.shape[:2]
    xc = xsize // 2 + 1
    yc = ysize // 2 + 1

    # Init boundaries and masks
    bd = calBoundary([xcR, ycR], [xsize, ysize])
    normMask = torch.zeros_like(maskRecons, device='cuda')
    maskRecons[xcR - 1, ycR - 1] = 1
    k_illu_cpu = k_illu_BF.copy().astype(np.float32)
    X, Y = torch.meshgrid(torch.arange(1, ysize + 1, device='cuda'),
                          torch.arange(1, xsize + 1, device='cuda'), indexing='ij')
    # Loop over NA-matching positions
    nNAmatching = recFTframe.shape[2]
    for idx in range(nNAmatching):
        bd2use = bd - np.tile(np.round(k_illu_cpu[idx, :]), (2, 1)).astype(int)
        maskOneside = (-(X - xc - k_illu_cpu[idx, 0]) * k_illu_cpu[idx, 0] -
                       (Y - yc - k_illu_cpu[idx, 1]) * k_illu_cpu[idx, 1] >
                       -0.5 * torch.norm(torch.tensor(k_illu_cpu[idx, :])))
        mask2useNew = mask2use * maskOneside
        unknownMask = (1 - maskRecons[bd2use[0, 0]-1:bd2use[1, 0], bd2use[0, 1]-1:bd2use[1, 1]]) * mask2use
        maskRecons[bd2use[0, 0]-1:bd2use[1, 0], bd2use[0, 1]-1:bd2use[1, 1]] += unknownMask
        offsetPhase = torch.angle(CTF_abe_cuda[xc + round(k_illu_cpu[idx, 0]) - 1,
                                               yc + round(k_illu_cpu[idx, 1]) - 1])
        normMask[bd2use[0, 0]-1:bd2use[1, 0], bd2use[0, 1]-1:bd2use[1, 1]] += mask2useNew
        if useAbeCorrection:
            ftRecons[bd2use[0, 0]-1:bd2use[1, 0], bd2use[0, 1]-1:bd2use[1, 1]] += (
                recFTframe[:, :, idx] * torch.conj(CTF_abe_cuda) *
                torch.exp(1j * offsetPhase) / (CTF_abe_abs_cuda + 1e-3))
        else:
            ftRecons[bd2use[0, 0]-1:bd2use[1, 0], bd2use[0, 1]-1:bd2use[1, 1]] += (
                recFTframe[:, :, idx] * mask2use / (mask2use + 1e-3))
    # Normalize and return spatial reconstruction
    normMask[xcR - 1, ycR - 1] = nNAmatching
    ftRecons = ftRecons * (normMask > 0.5) / (normMask + 1e-5) * maskRecons
    return ftRecons
