import os
import time
import argparse
import h5py

import matplotlib.pyplot as plt
from PIL import Image

import numpy as np
from scipy.fft import fft2, ifft2, fftshift, ifftshift

import torch
from subfunctionAPIC.calBoundary import calBoundary
from subfunctionAPIC.recFieldKK import recFieldKK
from subfunctionAPIC.findAbeFromOverlap import findAbeFromOverlap
from subfunctionAPIC.spectraStitching import spectraStitching
from subfunctionAPIC.recFieldFPMMultiplexing import recFieldFPMMultiplexing
from subfunctionAPIC.cNeoAlbedo import cNeoAlbedo


def main():
    # Options for the reconstruction
    parser = argparse.ArgumentParser()

    # General options
    parser.add_argument("--enableROI", default=True, type=bool, help="whether to use ROI in the reconstruction, this is the reconstruction size used in the code")
    parser.add_argument("--ROILength", default=512, type=int, help="define the ROI of the reconstruction")
    parser.add_argument("--ROIcenter", default='auto', type=str, help="define the center of ROI. Example: ROIcenter = [256,256]; ROIcenter = 'auto';")
    parser.add_argument("--paddingHighRes", default=4, type=int, help="define the upsampling ratio for the final high-res image")

    # Options for data loading
    parser.add_argument("--folderName", default='Data/experiment', type=str, help="Folder name, note this is case sensitive")
    parser.add_argument("--fileNameKeyword", default='NSCLC_02', type=str, help="e.g.: Siemens HEpath Thyroid")

    # Options for KK reconstruction
    parser.add_argument("--KK_wiener", default=True, type=bool, help="whether to use Wiener filter to mitigate noise")
    parser.add_argument("--KK_norm", default=True, type=bool, help="whether to normalize the acquired images such that they have the same effective intensity (mean value)")
    parser.add_argument("--KK_pad", default=3, type=int, help="Choose the zero-padding factor of the FT of the images")

    # Options for aberration correction
    parser.add_argument("--useAbeCorrection", default=True, type=bool, help="whether to use aberration correction")
    parser.add_argument("--Abe_weighted", default=True, type=bool, help="whether to use weighted matrix in the algorithm, in which case the algorithm focuses more on larger signals")

    # Options for darkfield reconstruction using multiplexed FPM
    parser.add_argument("--alpha", type=float, default= 1, help="Object update step-size parameter, should be very close to 1")
    parser.add_argument("--gamma", type=float, default= 0.01, help="regularization parameter")
    parser.add_argument("--noise_th", type=float, default=0.5, help="noise threshold for CTF")
    parser.add_argument("--maxIter", type=int, default=10, help="Maximum iteration")
    parser.add_argument("--minIter", type=int, default=1, help="Minimum iteration")
    parser.add_argument("--IC", type=int, default=0, help="LED intensity correction flag")
    parser.add_argument("--iter_IC", type=int, default=0, help="Which iteration to start intensity correction")
    parser.add_argument("--tol", type=float, default=0.1, help="Error tolerance")
    parser.add_argument("--virtualizeFPM", type=bool, default=False, help="When to show iterative results")
    parser.add_argument("--monotone", type=bool, default=True, help="whether to use monotone convergence in the algorithm")


    # Options for saving and visualization
    parser.add_argument("--saveResult",default=False,type=bool, help="whether to save the reconstruction results")
    parser.add_argument("--visualizeSumIm", default=True, type=bool, help="whether to visualize the sum of all measurements for comparison")
    parser.add_argument("--visualizeNAmatchingMeas", default=False, type=bool,help="whether to visualize the result using only NA-matching measurements")
    parser.add_argument("--visualizePupil", default=True, type=bool,help="whether to visualize the reconstructed pupil")


    args = parser.parse_args()
    saveResult = args.saveResult
    useAbeCorrection = args.useAbeCorrection
    enableROI = args.enableROI
    ROILength = args.ROILength
    ROIcenter = args.ROIcenter
    paddingHighRes = args.paddingHighRes
    visualizeSumIm = args.visualizeSumIm
    folderName = args.folderName
    fileNameKeyword = args.fileNameKeyword


    KK_wiener = args.KK_wiener
    KK_norm = args.KK_norm
    KK_pad = args.KK_pad
    Abe_weighted = args.Abe_weighted
    alpha = args.alpha
    gamma = args.gamma
    noise_th = args.noise_th
    IC = args.IC
    iter_IC = args.iter_IC
    tol = args.tol
    maxIter= args.maxIter
    minIter= args.minIter
    virtualizeFPM= args.virtualizeFPM
    monotone = args.monotone


    if not os.path.exists(folderName):  # Check if the folder exists
        raise FileNotFoundError(f"No folder with name '{folderName}' under current directory.")
    filename = os.path.join(folderName, f"{fileNameKeyword}.mat")
    with h5py.File(filename, 'r') as data_cal:
        I_low = data_cal['I_low'][:]
        freqXY_calib_BF = data_cal['freqXY_calib_BF'][:]
        cell_refs = data_cal['freqXY_calib_DF'][:]
        freqXY_calib_DF = [np.array(data_cal[ref]).transpose(1, 0) for ref in cell_refs.flatten()]
        mag = data_cal['mag'][0, 0]
        na_obj = data_cal['na_obj'][0, 0]
        na_illu = data_cal['na_illu'][0, 0]
        lambda_g = data_cal['lambda_g'][0, 0]
        na_rp_cal = data_cal['na_rp_cal'][0, 0]

    freqXY_calib_BF = freqXY_calib_BF.transpose(1, 0)
    I_low = I_low.transpose(2, 1, 0).astype(np.float32)
    nNAmatching = freqXY_calib_BF.shape[0]



    ## ======================================= Preprocessing for reconstruction =========================================
    ## Select measurement whose illumination NA matches up with the objective NA
    xsize, ysize = I_low.shape[0:2]
    xc = np.floor(xsize / 2 + 1)
    yc = np.floor(ysize / 2 + 1)

    if ROILength > xsize or ROILength > ysize:
        raise ValueError(f"ROI length cannot exceed {min(xsize, ysize)}")
    # Get the calibrated illumination angles for NA-matching measurements
    x_illumination_BF = freqXY_calib_BF[:, 1] - xc
    y_illumination_BF = freqXY_calib_BF[:, 0] - yc
    NA_pixel = na_rp_cal  # Calibrated maximum spatial freq in FT space
    print(f"Number of NA-matching measurements found: {nNAmatching}")


    # LED illumination angle, darkfield measurements
    x_illumination_DF = [freqXY_calib_DF_item[:,1]-xc for freqXY_calib_DF_item in freqXY_calib_DF]
    y_illumination_DF = [freqXY_calib_DF_item[:,0]-yc for freqXY_calib_DF_item in freqXY_calib_DF]

    # Change center to where the zero frequency is

    # Scaling x_illumination, y_illumination, and NA_pixel based on ROILength if enableROI is True
    if enableROI:
        x_illumination_BF *= ROILength / xsize
        y_illumination_BF *= ROILength / ysize
        x_illumination_DF = [x_illumination_DF_item * ROILength / xsize for x_illumination_DF_item in x_illumination_DF]
        y_illumination_DF = [y_illumination_DF_item * ROILength / ysize for y_illumination_DF_item in y_illumination_DF]
        NA_pixel *= ROILength / xsize  # Assuming NA_pixel is defined

        # Depending on the type of ROIcenter, calculate the boundary using calBoundary function
        if isinstance(ROIcenter, (list, np.ndarray)):
            bdROI = calBoundary(ROIcenter, ROILength)
        elif ROIcenter.lower() == 'auto':
            bdROI = calBoundary([xc, yc], ROILength)  # ROI locates in the center of the image
        else:
            raise ValueError("ROIcenter should be a 1-by-2 vector or 'auto'.")

        # Check if the boundary exceeds the image size
        if (bdROI < 1).any() or bdROI[0, 1] > xsize or bdROI[1, 1] > ysize:
            raise ValueError("ROI exceeds the boundary. Please check ROI's center and length")

        # Update xsize and ysize to ROILength
        xsize = ysize = ROILength
    else:
        # By default, use the maximum ROI
        bdROI = np.array([1, 1, xsize, ysize])

    if visualizeSumIm:
        I_sum = np.sum(I_low[bdROI[0, 0] - 1:bdROI[1, 0], bdROI[0, 1] - 1:bdROI[1, 1], :], axis=2)

    # Selecting a subset of I_low based on bdROI and indices slt_idx, slt_idxDF. Using numpy's advanced indexing
    I = I_low[bdROI[0, 0] - 1:bdROI[1, 0], bdROI[0, 1] - 1:bdROI[1, 1], :]

    del I_low

    ## Preparing for Reconstruction
    # Order measurement under NA-matching angle illumination
    theta = np.arctan2(y_illumination_BF, x_illumination_BF)
    pupilR = np.sqrt(x_illumination_BF ** 2 + y_illumination_BF ** 2)
    idxMap = np.argsort(theta)

    # Calculate Maximum Spatial Frequency
    enlargeF = 4
    Y, X = np.meshgrid(range(1, ysize * enlargeF + 1), range(1, xsize * enlargeF + 1))
    xc = xsize * enlargeF // 2 + 1
    yc = ysize * enlargeF // 2 + 1
    R_enlarge = np.abs(X - xc + 1j * (Y - yc))
    Y, X = np.meshgrid(range(1, ysize + 1), range(1, xsize + 1))
    xc = xsize // 2 + 1
    yc = ysize // 2 + 1
    R = np.abs(X - xc + 1j * (Y - yc))
    pupilRadius = max([NA_pixel, np.max(pupilR), np.linalg.norm(np.fix(np.column_stack((x_illumination_BF, y_illumination_BF))), axis=1).max()])
    CTF_Unresized = (R_enlarge < pupilRadius * enlargeF).astype('float32')
    im = Image.fromarray(CTF_Unresized)
    CTF = np.array(im.resize((xsize, ysize), Image.BILINEAR))
    CTF = np.maximum(np.roll(np.rot90(CTF, 2), (xsize % 2, ysize % 2), axis=(0, 1)), CTF)
    binaryMask = R <= 2 * pupilRadius


    # Noise Level Calculation and Image Stack Generation
    k_illu_BF = np.column_stack((x_illumination_BF[idxMap], y_illumination_BF[idxMap]))
    imStack_BF = np.zeros((I.shape[0], I.shape[1], nNAmatching), dtype=I.dtype)
    noiseLevel = np.zeros(nNAmatching)
    for idx in range(nNAmatching):
        ftTemp = fftshift(fft2(I[:, :, idxMap[idx]]))
        noiseLevel[idx] = max([np.finfo(float).eps, np.mean(np.abs(ftTemp[~binaryMask]))])
        ftTemp *= np.abs(ftTemp) / (np.abs(ftTemp) + noiseLevel[idx])
        imStack_BF[:, :, idx] = np.abs(ifft2(ifftshift(ftTemp * binaryMask)))

    k_illu_Dark = [np.round(np.column_stack((x_illumination_DF[i], y_illumination_DF[i]))) for i in range(len(x_illumination_DF))]
    imStack_Dark = I[:,:,nNAmatching:]  # This is the NA-matching measurements, which is used





    ## ======================================= Reconstruction starts there =========================================

    imsizeRecons = paddingHighRes * xsize
    ftRecons = torch.zeros(imsizeRecons, imsizeRecons, dtype=torch.complex64).cuda()
    maskRecons = torch.zeros(imsizeRecons, imsizeRecons, dtype=torch.float32).cuda()

    ## ============================ KK field reconstruction of NA-matching angle measurements  ============================
    # Convert to GPU for KK reconstruction
    imStack_cuda = torch.tensor(imStack_BF.copy(),dtype = torch.float32,device='cuda')
    k_illu_cpu = k_illu_BF.copy().astype(np.float32)
    CTF_cuda = torch.tensor(CTF.copy(),dtype=torch.float32,device='cuda')
    # ------------------------------------- KK reconstruction starts  -------------------------------------
    timestart = time.time()
    recFTframe, mask2use = recFieldKK(imStack_cuda, k_illu_cpu, ctf=CTF_cuda, pad=KK_pad, norm=KK_norm, wiener=KK_wiener)
                            # recFTframe: reconstructed complex spectrums of NA-matching measurements
    timeKK = time.time()
    print('KK Reconstruction Time: ', timeKK-timestart)




    ## ============================ Aberration estimation from KK spectra ============================
    # Convert back to numpy arrays for the findAbeFromOverlap function
    k_illu_cpu = np.round(k_illu_BF.copy()).astype(np.int32)
    recFTframe_cpu = recFTframe.cpu().numpy().astype(np.complex64)
    # ------------------------------------- Aberration correction starts  -------------------------------------
    CTF_abe, zernikeCoeff = findAbeFromOverlap(recFTframe_cpu, k_illu_cpu, CTF, weighted=Abe_weighted)
    timeFindAbe = time.time()
    print('Aberration Estimation Time: ', timeFindAbe-timeKK)




    ## ============================ KK spectra stitching and aberration correction ============================
    if useAbeCorrection:
        CTF_abe_cuda = torch.tensor(CTF_abe.astype(np.complex64), device='cuda')
    else:
        CTF_abe_cuda = CTF_cuda
    # -------------------------------------  KK spectra stitching and aberration correction starts  -------------------------------------
    ftRecons = spectraStitching(ftRecons, maskRecons, recFTframe, CTF_abe_cuda, mask2use, k_illu_BF, useAbeCorrection)
    himMatching = torch.fft.ifft2(torch.fft.ifftshift(ftRecons, dim=[-2, -1]), dim=[-2, -1])
    timeFindAbe_Finish = time.time()
    print('Abe Subtraction Time: ', timeFindAbe_Finish-timeFindAbe)


    ## ============================ Darkfield reconstruction using multiplexed FPM ============================
    imStack_Dark = torch.tensor(imStack_Dark,dtype=torch.float32, device='cuda')
    O0 = ftRecons
    P0 = CTF_abe_cuda
    H0 = torch.tensor(CTF, device='cuda')
    timeFPMMultiplexing = time.time()
    opts = {
        "imsizeRecons" : imsizeRecons,
        "O0":O0,
        "P0":P0,
        "H0": H0,
        "tol":tol,
        "maxIter":maxIter,
        "minIter":minIter,
        "virtualizeFPM":virtualizeFPM,
        "alpha": alpha,
        "gamma": gamma,
        "noise_th":noise_th,
        "ftMatching_Mask":maskRecons.to(torch.bool),
        "IC": IC,
        "iter_IC": iter_IC,
        "monotone" : monotone
    }

    O = recFieldFPMMultiplexing(imStack_Dark,k_illu_Dark,opts)
    timeend = time.time()
    print('DF Time: ', timeend-timeFPMMultiplexing)
    print('Final_Time: ', timeend-timestart)
    ## ======================================= Reconstruction ends =========================================





    ## Visualization and saving results
    if saveResult:
        import scipy.io as sio
        sio.savemat(fileNameKeyword+'_Results.mat', {'O':O.cpu().numpy(),'CTF_abe':CTF_abe})

    Nx = 1536
    O_crop = crop_center(O, Nx)
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(np.abs(O_crop), cmap='gray')
    plt.title("APIC — Amplitude")
    plt.axis('off')
    plt.subplot(1, 2, 2)
    plt.imshow(np.angle(O_crop), cmap='pink')
    plt.title("APIC — Phase")
    plt.axis('off')
    plt.tight_layout()
    plt.show()


    if args.visualizeNAmatchingMeas:
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        plt.imshow(np.abs(crop_center(himMatching, Nx)), cmap='gray')
        plt.title("NA-Matching — Amplitude")
        plt.axis('off')

        plt.subplot(1, 2, 2)
        fft_img = np.log1p(np.abs(np.fft.fftshift(np.fft.fft2(himMatching.cpu().numpy()))))
        plt.imshow(fft_img, cmap='gray')
        plt.title("NA-Matching — Spectrum")
        plt.axis('off')
        plt.tight_layout()
        plt.show()

    if args.visualizePupil:
        plt.figure()
        cNeoAlbedo_colormap = cNeoAlbedo()
        pupil_amp_mask = (np.abs(CTF_abe) > 1e-3)
        plt.imshow(np.angle(CTF_abe) * pupil_amp_mask,
                   cmap=cNeoAlbedo_colormap, vmin=-np.pi, vmax=np.pi)
        plt.axis('off')
        plt.title('Reconstructed Pupil — APIC')
        plt.colorbar()
        if args.saveResult:
            os.makedirs('Results', exist_ok=True)
            plt.savefig(os.path.join('Results', 'Pupil_Results.png'), dpi=300)
        plt.show()



def crop_center(img, L=200):
    if isinstance(img, torch.Tensor):
        img = img.cpu().numpy()
    H, W = img.shape
    cx, cy = W // 2, H // 2
    hL = L // 2
    return img[cy-hL:cy+hL, cx-hL:cx+hL]

if __name__ == "__main__":

    main()


