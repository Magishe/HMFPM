import numpy as np
import torch
from subfunctionAPIC.objectUpdate import objectUpdate
from subfunctionAPIC.visualizeFPMResults import visualizeFPMResults
import matplotlib.pyplot as plt

def recFieldFPMMultiplexing(imStack_Dark,k_illu_Dark,opts):
    """
    Reconstruct high-resolution complex object using multiplexed FPM data with darkfield illumination

    Parameters:
    I: Multiplexed intensity measurements by different LEDs (numpy array)
    k_illu_Dark: illumination wavevector
    opts: Dictionary of options for the algorithm

    Returns:
    O: Reconstructed high-resolution complex object
    """
    Nmy, Nmx, Nimg = imStack_Dark.shape
    Np = (Nmy, Nmx)
    cen0 = np.round((np.array(opts['imsizeRecons']).astype(np.float32) + 2) / 2).astype(int)

    # Initialize variables
    H0 = torch.abs(torch.conj(opts['H0'])*opts['H0']) > opts["noise_th"]
    print('| iter |  rmse    |')
    print('-' * 20)
    for j in range(1, 21):
        print('-', end='')
    print()

    ## Initialization in FT domian
    P = opts['P0']
    P_abs2 = torch.abs(torch.conj(P)*P)
    P_mask = torch.sqrt(torch.max(P_abs2))
    opts['P0'] = 0
    O = opts['O0']

    O0 = opts['O0'].clone()
    ftRecons_Mask = opts["ftMatching_Mask"]


    err1 = float('inf')
    err2 = 50.0
    err = []
    iter = 0

    while np.abs(err1 - err2) > opts['tol'] and iter < opts['maxIter']:
        err1 = err2
        err2 = 0.0
        iter += 1
        updating_full_mask = opts["ftMatching_Mask"].clone()
        for m in range(Nimg):
            Ns_temp = k_illu_Dark[m]
            r0 = Ns_temp.shape[0]
            Psi0 = torch.zeros((Np[0], Np[1], r0),dtype=torch.complex64,device='cuda')
            updating_mask_batch = torch.zeros((Np[0], Np[1], r0), dtype=torch.bool, device='cuda')
            cen = np.zeros((2, r0),dtype=np.int32)
            for p in range(r0):
                cen[:, p] = cen0 - Ns_temp[p, :].ravel()-1
                A = downsamp(O, cen[:, p], Np)
                updating_mask_batch[:,:,p] = downsamp(updating_full_mask, cen[:, p], Np)
                Psi0[:, :, p] = A* P * H0
            psi0 = Ft(Psi0)
            I_est = torch.mean(torch.abs(torch.conj(psi0)*psi0), dim=2, dtype=torch.float32)
            I_mea = imStack_Dark[:, :, m]
            if opts['IC'] and iter >= opts['iter_IC']:
                tt = torch.mean(I_est) / torch.mean(torch.abs(I_mea))
                I_mea = I_mea * tt
            dPsi = F(torch.sqrt(I_mea).unsqueeze(-1) * psi0 / (torch.sqrt(I_est).unsqueeze(-1) + torch.finfo(I_est.dtype).eps)) * H0.unsqueeze(-1)
            dPsi_temp = dPsi / torch.mean(torch.abs(dPsi*updating_mask_batch)) * torch.mean(torch.abs(Psi0*updating_mask_batch))

            O,updating_full_mask = objectUpdate(
                O, P, dPsi_temp, cen,
                opts["alpha"],
                opts["gamma"],
                opts["ftMatching_Mask"],
                opts["noise_th"],
                P_abs2,
                P_mask,
                updating_full_mask
            )
            err2 = err2 + torch.sqrt(torch.sum((I_mea - I_est) ** 2)).cpu()
        err.append(err2)

        # Visualization
        if opts['virtualizeFPM']:
            visualizeFPMResults(O, P, iter)
        print(f'| {iter:2d}   | {err2:.6e} |')

        # Monotonicity check: use the result from the last iteration if error increases
        if opts.get('monotone', False) and iter > opts.get('minIter', 0):
            if err2 > err1:
                O = O_bef
                break
        O[ftRecons_Mask] = O0[ftRecons_Mask]
        O_bef = O

    O = Ft(O)
    return O



def downsamp(x, cen, Np):
    """    Downsamping function for spectra cropping """
    return x[round(cen[0] - Np[0] // 2): round(cen[0] - Np[0] // 2 + Np[0]),
             round(cen[1] - Np[1] // 2): round(cen[1] - Np[1] // 2 + Np[1])]

def F(x):
    """    Forward Fourier transform function """
    return torch.fft.fftshift(torch.fft.fft2(x, dim=(0, 1)), dim=(0, 1))

def Ft(x):
    """    Inverse Fourier transform function """
    return torch.fft.ifft2(torch.fft.ifftshift(x, dim=(0, 1)), dim=(0, 1))

