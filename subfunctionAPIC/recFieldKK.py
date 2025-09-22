import numpy as np
from subfunctionAPIC.calBoundary import calBoundary
import warnings
import torch
def recFieldKK(imStack, mycoord, **kwargs):
    """
        use kramers-kronig to do field reconstruction
        Requried inputs:
          1. imStack: The image acquired under critial-angle illuminations
          2. mycoord: Coordinate of the illumination vector in terms of pixel in
              the spatial frequency space. i.e. the pixel-wise shift of the
              zero-frequency after Fourier transform (FT).
        Optional arguments:
          1. CTF: enable CTF in the reconstruction. When enabled, the
              reconstructed field whose spatial frequency lies outside the CTF
              support will be set to zero.
          2. Support: the support of the image's FT. When used, we only keep the
              image's FT within the support (To reduce noise).
          3. Normalizaiton: whether to normalize the acquired images such that
              they have the same effective intensity (mean value). Disabled by
              default.
          4. Padding: Choose the zero-padding factor of the FT of the images.
              Default padding factor is 3. This should integer that is larger
              than 1.
          5. Wiener: whether to use Wiener filter to mitigate noise. Disabled by
              default.
          6. noise floor: set the noise floor of the acquired image. It helps to
              prevent image from showing weired phases in the dark region where
              the SNR is too low for the algorithm to extract the right phases.
              The noise floor is set automatically when no number is specified.
          7. use data intensity: treat the square root of measured intensity as
              the ground truth for amplitude. This is enabled by default.

        By Ruizhi Cao, Dec 7, 2022, Biophotonics Lab, Caltech
        Implemented in Python by Shi Zhao, Haowen Zhou and Steven(Siyu) Lin on Nov 16, 2023
    """
    # Default values for optional parameters
    padfactor = 3
    useCTF = False
    useSupport = False
    normIntensity = False
    useNoiseFilter = False
    autoNoiseFloor = True
    useDataIntensity = True

    # Parse optional arguments
    for key, value in kwargs.items():
        key = key.lower()
        if key == 'ctf':
            CTF = value
            useCTF = True
        elif key in {'support', 'otf', 'mask'}:
            maskFilt = value
            useSupport = True
        elif key in {'normalization', 'norm'}:
            normIntensity = value
        elif key in {'pad', 'padding', 'padfactor', 'factor'}:
            padfactor = value
        elif key in {'wiener', 'reg', 'regularization', 'wiener filer', 'wienerfilter'}:
            useNoiseFilter = value
        elif key in {'noise floor', 'noisefloor', 'noise'}:
            regImage = value
            autoNoiseFloor = False
            if regImage <= 0:
                autoNoiseFloor = True
                warnings.warn('Warning: Noise level must be positive. Program uses its default noise level.')
        elif key in {'data intensity', 'replace intensity', 'use data intensity'}:
            useDataIntensity = (value != 0)
        else:
            raise ValueError(
                "Supported options are 'pad', 'normalization', 'wiener', 'noise floor', 'CTF', and 'support'.")

    # Check if padfactor is an integer, if not, round up
    if padfactor % 1 != 0:
        warnings.warn('Warning: Padding factor must be an integer. It is rounded up.')
        padfactor = np.ceil(padfactor)

    # Check if padfactor is less than 1
    if padfactor < 1:
        raise ValueError('Pad factor must be larger than 1.')

    # Check if noise filter is used without CTF
    if useNoiseFilter and not useCTF:
        raise ValueError('To suppress noise, CTF must be given to the program to estimate noise\'s amplitude.')

    # Get dimensions and calculate center
    xsize, ysize, nBrightField = imStack.shape
    xc = xsize // 2 + 1
    yc = ysize // 2 + 1

    # Normalize the effective intensity of each image
    if normIntensity:
        meanIFrame_torch = torch.mean(imStack, dim=[0,1])
        meanIFrame_torch /= torch.mean(meanIFrame_torch)

        # Perform the division for each slice in the stack
        for idx in range(nBrightField):
            imStack[:, :, idx] /= meanIFrame_torch[idx]

    # Assign spaces for the padded images and FT array
    paddedFT = torch.zeros((nBrightField, xsize * padfactor, ysize * padfactor), dtype=torch.complex64, device='cuda')
    imStackPad = torch.zeros((nBrightField, xsize * padfactor, ysize * padfactor), device='cuda')

    xcpad = xsize * padfactor // 2 + 1
    ycpad = ysize * padfactor // 2 + 1

    # Assuming calBoundary is a function you have defined to calculate the boundary
    bdpad = calBoundary([xcpad, ycpad], [xsize, ysize])

    # Parallel implementation of the for loop
    ftIm_batch = torch.fft.fftshift(torch.fft.fft2(imStack.permute(2, 0, 1), dim=[-2, -1]), dim=[-2, -1])
    if useSupport:
        paddedFT[:,bdpad[0,0]-1:bdpad[1,0], bdpad[0,1]-1:bdpad[1,1]] = maskFilt * ftIm_batch
    else:
        paddedFT[:,bdpad[0,0]-1:bdpad[1,0], bdpad[0,1]-1:bdpad[1,1]] = ftIm_batch
    imStackPad = torch.real(torch.fft.ifft2(torch.fft.ifftshift(paddedFT, dim=[-2, -1]), dim=[-2, -1]))
    imStackPad = imStackPad * (imStackPad >= 0)
    imStackPad = imStackPad.permute(1, 2, 0)



    recFTframe = torch.zeros((xsize, ysize, nBrightField), dtype=torch.complex64, device='cuda')

    imSum = torch.mean(imStackPad[:, :, :nBrightField], dim=2)
    maxSig = torch.max(imSum)

    # Set the noise floor if autoNoiseFloor is True
    if autoNoiseFloor:
        regImage = min(1 / padfactor, maxSig / 10000)  # 65535 for a 16-bit camera

    # Create meshgrid (I think torch and np are just inverse of X and Y)
    X, Y = torch.meshgrid(torch.arange(1, ysize * padfactor + 1, device='cuda'),
                          torch.arange(1, xsize * padfactor + 1, device='cuda'),indexing = 'ij')



    # Build Wiener filters
    if useNoiseFilter:
        wienerPixelTol = 3
        wienerRingWidth = 5
        R = torch.abs(
            X[bdpad[0,0]-1:bdpad[1,0], bdpad[0,1]-1:bdpad[1,1]] - xcpad + 1j * (Y[bdpad[0,0]-1:bdpad[1,0], bdpad[0,1]-1:bdpad[1,1]] - ycpad))
        CTFmax = torch.sqrt(torch.sum(CTF) / torch.pi)
        wienerMask = (R > (CTFmax + wienerPixelTol)) & (R < (CTFmax + wienerPixelTol + wienerRingWidth))

        # Check if CTF is a logical array
        if CTF.dtype == torch.bool:
            wienerMask = wienerMask & ~CTF
        else:
            wienerMask = wienerMask & (torch.abs(CTF) < 1e-2)

    # Create mask to use
    mask2use = (CTF > 0.9).float()


    for idx in range(nBrightField):
        realPart = torch.log(imStackPad[:, :, idx] + regImage)
        tempFT = torch.fft.fftshift(torch.fft.fft2(realPart, dim=[-2, -1]), dim=[-2, -1])
        maskOneside = (-(X - xcpad) * mycoord[idx, 0] - (Y - ycpad) * mycoord[idx, 1] > 0.5 * torch.norm(torch.tensor(mycoord[idx, :])))
        temp = tempFT * maskOneside

        recField = torch.exp(0.5 * realPart + 1j * torch.imag(torch.fft.ifft2(torch.fft.ifftshift(temp))))

        # Round mycoord for use as indices
        shift_values = tuple(np.round(mycoord[idx, :]).astype('int')) # Convert shift values to integers
        mycoord[idx, :] = np.round(mycoord[idx, :])
        temp = torch.roll(torch.fft.fftshift(torch.fft.fft2(recField)) / padfactor, shift_values, dims=(0, 1))

        if useCTF:
            zeroFreqWeight = temp[xcpad + int(mycoord[idx, 0])-1, ycpad + int(mycoord[idx, 1])-1]
            # if CTF[xc + int(mycoord[idx, 0])-1, yc + int(mycoord[idx, 1])-1] < 1e-2:
            #     raise ValueError(
            #         'Peak corresponds to the zero-frequency cropped out using current CTF. Please increase the CTF.')

            if useNoiseFilter:
                shifted_R = torch.roll(R, shifts=tuple(mycoord[idx, :].astype(int)), dims=(0, 1))
                wienerFilterTemp = maskOneside[bdpad[0,0] - int(mycoord[idx, 0])-1:bdpad[1,0] - int(mycoord[idx, 0]),
                                   bdpad[0,1] - int(mycoord[idx, 1])-1:bdpad[1,1] - int(mycoord[idx, 1])] & wienerMask
                wienerFilterTemp = wienerFilterTemp & (shifted_R < 2 * CTFmax)
                noiseReg = 0.2 * torch.sum(torch.abs(temp[bdpad[0, 0] - 1:bdpad[1, 0], bdpad[0, 1] - 1:bdpad[1, 1]]) * wienerFilterTemp) / torch.sum(wienerFilterTemp)
                recFTframe[:, :, idx] = temp[bdpad[0, 0] - 1:bdpad[1, 0], bdpad[0, 1] - 1:bdpad[1, 1]] * mask2use / (CTF + 1e-3)
                recFTframe[:, :, idx] *= torch.abs(recFTframe[:, :, idx]) / (torch.abs(recFTframe[:, :, idx]) + noiseReg)

            else:
                recFTframe[:, :, idx] = temp[bdpad[0, 0] - 1:bdpad[1, 0], bdpad[0, 1] - 1:bdpad[1, 1]] * mask2use / (CTF + 1e-3)

            recFTframe[xc + int(mycoord[idx, 0]) - 1, yc + int(mycoord[idx, 1]) - 1, idx] = zeroFreqWeight

        else:
            recFTframe[:, :, idx] = temp[bdpad[0,0]-1:bdpad[1,0], bdpad[0,1]-1:bdpad[1,1]]

        if useDataIntensity:
            temp = torch.fft.ifft2(torch.fft.ifftshift(recFTframe[:, :, idx],dim=[-2, -1]),dim=[-2, -1])

            # Multiply by sqrt of imStack (adding 0j for complex) and multiply by exp(1j * angle of temp)
            recFTframe[:, :, idx] = torch.fft.fftshift(torch.fft.fft2(torch.sqrt(imStack[:, :, idx] * (1 + 0j)) * torch.exp(1j * torch.angle(temp)),dim=[-2, -1]),dim=[-2, -1])
    return recFTframe, mask2use


