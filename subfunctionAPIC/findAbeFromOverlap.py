import warnings
import numpy as np
from subfunctionAPIC.zernike.zernfun2 import zernfun2
from subfunctionAPIC.calBoundary import calBoundary
from subfunctionAPIC.phase_unwrapCG import phase_unwrapCG
from scipy.sparse import vstack
from PIL import Image
from scipy.ndimage import zoom
from scipy.signal import medfilt2d
from numpy import ceil
from scipy.sparse import csr_matrix
import cv2
from skimage.restoration import unwrap_phase

def findAbeFromOverlap(recFTframe, mycoord, CTF, **kwargs):
    """
      findAbeFromOverlap Determine aberration from multiple KK reconstructions with different illumination angles
      This function estimate the aberration from the image-level KK
      reconstruction. Evenly spaced LEDs along the ring, together with more
      images, give a better result.
      Options
          1. zernike mode: zernike mode to optimize for abeeration estimation
          2. use image: select which image to use in the algorithm.
          3. weighted: use weighted matrix in the algorithm, in which case
                       the algorithm focuses more on larger signals.
          4. max CTF: define the maximal spatial frequency of the CTF
                      manually. Use this for a more accurate (subpixel)
                      cutoff frequency estimation.
          5. closest n pairs: for a given spectrum, pair it with the closest
                              n other spectrums and extract their overlaps
                              for aberration correction. By default, the
                              program finds the closest 1 pair.

    """
    # Default parameters
    xsize, ysize, nBrightField = recFTframe.shape
    useimg = np.arange(nBrightField)  # Based on the third dimension of recFTframe
    nPairs2use_eachSpectrum = 1
    zernikeMode2Opt = np.arange(3,36)  # Zernike modes to optimize
    marginPix = 2
    useWeights = True
    desiredMed2MaxWeightRatio = 2  # Ratio of weight of the median signal to that of the maximal signal
    mycoord = np.round(mycoord).astype('int')
    CTFThreshold = 1 * 10 ** (-2)

    # Parse optional arguments
    for key, value in kwargs.items():
        key = key.lower()
        if key in {'zernikemode', 'zernike mode'}:
            zernikeMode2Opt = value
        elif key in {'useimage', 'use image'}:
            useimg = value
        elif key in {'use weight', 'weighted', 'weight'}:
            useWeights = value
        elif key in {'max ctf', 'cutoff', 'max freq'}:
            maxCTFUser = value
        elif key in {'find overlap', 'find pairs', 'closest n pairs', 'n pairs'}:
            if isinstance(value, (int, float)) and value > 0:
                nPairs2use_eachSpectrum = value
            else:
                raise ValueError('Please specify the argument for "closest n pairs" with a natural number.')
        else:
            raise ValueError('Supported options are "use image", "weighted", "cutoff", "n pairs", and "zernike mode".')

    # Check for valid zernike modes
    if any(zernikeMode2Opt < 3):
        warnings.warn('Warning: The first 3 zernike mode is dropped, as those produce a worse estimation.')
        zernikeMode2Opt = zernikeMode2Opt[zernikeMode2Opt >= 3]

    # Validate useimg
    if any(useimg < 0) or any(useimg > nBrightField - 1):
        raise ValueError('The index of image must be positive, and should not be greater than the number of images.')
    if nPairs2use_eachSpectrum > round(nBrightField / 2):
        warnings.warn("The program is asked to pair one spectrum with more than half of the acquired spectrums. This "
                      "is likely to be wrong")

    # generate a library for the spectrum pairs
    useimg = np.sort(useimg)
    idxNextLib = np.zeros(len(useimg) * nPairs2use_eachSpectrum, dtype=int)
    idxTemp = np.arange(len(useimg))
    for idxPair in range(nPairs2use_eachSpectrum):
        idxNextLib[(idxTemp + 1) * nPairs2use_eachSpectrum - nPairs2use_eachSpectrum + (idxPair + 1) - 1] = np.mod(
            useimg + 1 + idxPair,
            nBrightField)
    useimg = np.tile(useimg, (nPairs2use_eachSpectrum, 1)).T.flatten()

    # Calculate the center coordinates
    xc = xsize // 2 + 1
    yc = ysize // 2 + 1
    nZernike = len(zernikeMode2Opt)
    maxCTF = np.argmax(np.abs(CTF[xc - 1, yc:]) < CTFThreshold) + 1  # The same as matlab, need to be revised in the index

    # Calculate dcAmp
    dcAmp = np.zeros(nBrightField)
    for idx in range(nBrightField):
        dcAmp[idx] = np.abs(recFTframe[int(xc + mycoord[idx, 0] - 1), int(yc + mycoord[idx, 1] - 1), idx])

    # Find the maximum DC amplitude
    maxDC = np.max(dcAmp)

    # Determine the resolution pixel size
    resPix = min(xsize, ysize) - 1 - 2 * maxCTF

    # Adjust margin pixel size if necessary
    if marginPix * 2 > resPix:
        marginPix = np.floor(resPix / 2)

    # Check for existence of maxCTFUser and compare with maxCTF
    if 'maxCTFUser' in globals():
        if maxCTF < maxCTFUser:
            warnings.warn('Warning: The thresholding-based cutoff frequency estimate is smaller than the given value. '
                          'The code will overwrite this value. Please check the given cutoff freq.')
            maxCTFUser = maxCTF
        temp = np.linspace(-maxCTF / maxCTFUser, maxCTF / maxCTFUser, 2 * maxCTF + 1)
    else:
        temp = np.linspace(-1, 1, 2 * maxCTF + 1)

    Yz, Xz = np.meshgrid(temp, temp)
    theta, r = np.arctan2(Yz, Xz), np.sqrt(Xz ** 2 + Yz ** 2)
    idx2use = r <= 1
    zernikeTemp = zernfun2(zernikeMode2Opt, r[idx2use], theta[idx2use])

    Hz = np.zeros(((2 * maxCTF + 1) ** 2, nZernike))

    # Populate the matrix with Zernike polynomials
    for idx in range(nZernike):
        Hz[idx2use.flatten(), idx] = zernikeTemp[:, idx]
    del zernikeTemp, theta, r

    # boundary of the 'OTF' (nonzero signal in FT of the intensity image)
    bd = calBoundary([xc, yc], [4 * maxCTF + 1, 4 * maxCTF + 1])
    Y, X = np.meshgrid(np.arange(1, 4 * maxCTF + 2), np.arange(1, 4 * maxCTF + 2))
    if isinstance(CTF, np.ndarray) and CTF.dtype == np.bool_:
        tempCTF = CTF[bd[0,0] - 1:bd[1,0], bd[0,1] - 1:bd[1,1]]
    else:
        tempCTF = CTF[bd[0,0] - 1:bd[1,0], bd[0,1] - 1:bd[1,1]] >= CTFThreshold
    bdsmall = calBoundary([2 * maxCTF + 1, 2 * maxCTF + 1], [2 * maxCTF + 1, 2 * maxCTF + 1])

    Hdiff = []  #operator that calculates the aberration differences note here Hdiff is basically D_{il} in our derivation
    first_sparse = 0
    phaseMeas = []
    ncolHdiff = (2 * maxCTF + 1) ** 2
    weightsVct = []
    numPhaseMeas = []
    offsetIdx = []
    countsIdxNext = 0

    for idx in useimg:
        idxNext = idxNextLib[countsIdxNext]
        countsIdxNext += 1
        relShift = mycoord[idx, :] - mycoord[idxNext, :]

        maskOneside1 = (-(X - 2 * maxCTF - 1 - mycoord[idx, 0]) * mycoord[idx, 0]
                        - (Y - 2 * maxCTF - 1 - mycoord[idx, 1]) * mycoord[idx, 1]
                        > -0.5 * np.linalg.norm(mycoord[idx, :]))
        maskOneside2 = (-(X - 2 * maxCTF - 1 - mycoord[idxNext, 0]) * mycoord[idxNext, 0]
                        - (Y - 2 * maxCTF - 1 - mycoord[idxNext, 1]) * mycoord[idxNext, 1]
                        > -0.5 * np.linalg.norm(mycoord[idxNext, :]))

        overlapCTF = (tempCTF & maskOneside1) & np.roll((tempCTF & maskOneside2), relShift.astype('int'), axis=(0, 1))
        overlapCTF2 = np.roll(overlapCTF, -relShift.astype('int'), axis=(0, 1))

        # index of the zero-freq in the vectorized spectrum
        originSpectrum1 = (maxCTF + mycoord[idx, 1]) * (2 * maxCTF + 1) + maxCTF + mycoord[idx, 0]
        originSpectrum2 = (maxCTF + mycoord[idxNext, 1]) * (2 * maxCTF + 1) + maxCTF + mycoord[idxNext, 0]

        posIdx = np.flatnonzero(overlapCTF[bdsmall[0,0] - 1:bdsmall[1,0], bdsmall[0,1] - 1:bdsmall[1,1]].T.flatten())
        negIdx = np.flatnonzero(overlapCTF2[bdsmall[0,0] - 1:bdsmall[1,0], bdsmall[0,1] - 1:bdsmall[1,1]].T.flatten())
        nMeasure = len(posIdx)

        phaseTemp = recFTframe[bd[0,0] - 1:bd[1,0], bd[0,1] - 1:bd[1,1], idx] * tempCTF * \
                    np.conj(np.roll(recFTframe[bd[0,0] - 1:bd[1,0], bd[0,1] - 1:bd[1,1], idxNext] * tempCTF, relShift.astype('int'),
                                    axis=(0, 1)))

        if useWeights:
            # Flatten phaseTemp as in MATLAB (column-major order)
            phaseTemp_flattened = np.ravel(phaseTemp, order='F')
            # Apply the boolean mask from overlapCTF (also flattened)
            weightsTemp = np.abs(phaseTemp_flattened[overlapCTF.ravel(order='F')])

            # reduce the phase disturbance due to non-continuous edge
            temp = np.full(overlapCTF.shape, False)
            temp[2 * maxCTF + mycoord[idx, 0], :] = True
            temp[:, 2 * maxCTF + mycoord[idx, 1]] = True
            temp[2 * maxCTF + mycoord[idx, 0], 2 * maxCTF + mycoord[idx, 1]] = False
            weightsTemp[temp.T[overlapCTF.T]] = 0

            weightsVct = np.append(weightsVct, weightsTemp*((weightsTemp>0).astype('int')))

        tempXCTF = np.flatnonzero(np.sum(overlapCTF, axis=1) >= 0.1)
        tempYCTF = np.flatnonzero(np.sum(overlapCTF, axis=0) >= 0.1)
        # Determine the smallest rectangle containing the overlapped region
        cropCoord = [min(tempXCTF), max(tempXCTF), min(tempYCTF), max(tempYCTF)]
        # Weights for phase unwrapping
        tempWeights = np.log10(np.abs(phaseTemp[cropCoord[0]:cropCoord[1] + 1, cropCoord[2]:cropCoord[3] + 1]) + 1)
        # Unwrap the phase before aberration extraction
        for i in range(phaseTemp.shape[0]):
            for j in range(phaseTemp.shape[1]):
                if abs(phaseTemp[i,j]) == 0:
                    phaseTemp[i, j] = 0
        # phaseUnwrapTemp = unwrap_phase(np.angle(phaseTemp[cropCoord[0]:cropCoord[1] + 1, cropCoord[2]:cropCoord[3] + 1]))
        phaseUnwrapTemp = phase_unwrapCG(np.angle(phaseTemp[cropCoord[0]:cropCoord[1] + 1, cropCoord[2]:cropCoord[3] + 1]),
                                            tempWeights / np.max(tempWeights))

        # Raw phase extraction
        phaseRaw = np.angle(phaseTemp[cropCoord[0]:cropCoord[1] + 1, cropCoord[2]:cropCoord[3] + 1])

        # Force phase unwrap using multiples of 2pi
        wrappedPhase = phaseUnwrapTemp - phaseRaw
        x2use = np.arange(-2, 2  + 1 / 8, 1 / 8) * np.pi
        N, _ = np.histogram(
            wrappedPhase[overlapCTF[cropCoord[0]:cropCoord[1] + 1, cropCoord[2]:cropCoord[3] + 1]].flatten(),
            bins=x2use)
        idx2use = (x2use >= -(np.pi + np.pi / 4)) & (x2use <= np.pi + np.pi / 4)
        N = np.append(N, 0)
        N[~idx2use] = 0
        idxPk = np.argmax(N)  # Use the first peak when multiple maxima are found.
        offsetPk = np.mean(
            wrappedPhase[(wrappedPhase >= (x2use[idxPk] - np.pi / 4)) & (wrappedPhase <= (x2use[idxPk] + np.pi / 4))])

        phaseTemp = phaseRaw + np.round((wrappedPhase - offsetPk) / (2 * np.pi)) * 2 * np.pi + offsetPk


        index_x = (2 * maxCTF + 1) + mycoord[idx, 0] - cropCoord[0] - 1
        index_y = (2 * maxCTF + 1) + mycoord[idx, 1] - cropCoord[2] - 1
        phaseTemp -= phaseTemp[index_x, index_y]

        # Update phaseMeas
        phaseMeas = np.concatenate([phaseMeas, phaseTemp.T[
            overlapCTF[cropCoord[0]:cropCoord[1] + 1, cropCoord[2]:cropCoord[3] + 1].T]])

        # Update Hdiff
        row_indices = np.tile(np.arange(nMeasure), 2)
        col_indices = np.concatenate([posIdx, negIdx])
        data = np.concatenate([np.ones(nMeasure), -np.ones(nMeasure)])
        new_row_sparse = csr_matrix((data, (row_indices, col_indices)), shape=(nMeasure, ncolHdiff))
        Hdiff = vstack((Hdiff, new_row_sparse)) if first_sparse != 0 else new_row_sparse


        # Append to numPhaseMeas
        numPhaseMeas = np.append(numPhaseMeas, nMeasure)

        # Update offsetIdx
        new_offset = np.array([[originSpectrum1, originSpectrum2]])
        offsetIdx = np.vstack([offsetIdx, new_offset]) if first_sparse != 0 else new_offset  # Index
        first_sparse += 1

    # Initialize Hoffset
    Hoffset = np.zeros((sum(numPhaseMeas).astype('int'), nZernike))
    # Populate Hoffset
    for idx in range(len(numPhaseMeas)):
        idxSt = 1 + int(sum(numPhaseMeas[:idx]))
        Hoffset[idxSt - 1:idxSt - 1 + numPhaseMeas[idx].astype('int'), :] = np.tile(Hz[offsetIdx[idx, 0] , :] - Hz[offsetIdx[idx, 1], :], (numPhaseMeas[idx].astype('int'), 1))
    # Apply Weights if useWeights is True
    if useWeights:
        medWeights = np.median(weightsVct)
        ratioMax2Med = (maxDC ** 2) / medWeights
        factor = np.ceil(desiredMed2MaxWeightRatio * np.log10(ratioMax2Med) / (desiredMed2MaxWeightRatio - 1))
        weightsVct = np.log10(weightsVct / np.max(weightsVct) * 10 ** factor + 1)
        Hoverall = weightsVct[:, np.newaxis] * (Hdiff @ Hz - Hoffset)
    else:
        Hoverall = Hdiff @ Hz - Hoffset
    del Hdiff, Hoffset




    ## compensate for noise related phase difference offset: (W*) (D*Z*x + H_e*e) = (W*)y; Herror = H_e'*H_e
    sigOffset = np.zeros(len(useimg))
    blockLL = np.zeros((len(useimg), nZernike))
    weightsVctSq = weightsVct ** 2  # Squared weights
    weightsSqSum = np.ones(len(useimg))

    # Loop over useimg
    for idx in range(len(useimg)):
        idxSt = int(sum(numPhaseMeas[:idx]))

        if useWeights:
            block_subset = Hoverall[idxSt:idxSt + int(numPhaseMeas[idx]) :]
            weights_subset = weightsVct[idxSt:idxSt +int(numPhaseMeas[idx])]
            weights_sq_subset = weightsVctSq[idxSt:idxSt + int(numPhaseMeas[idx])]

            blockLL[idx, :] = np.sum(weights_subset[:, np.newaxis] * block_subset, axis=0)
            weightsSqSum[idx] = np.sum(weights_sq_subset)
            sigOffset[idx] = np.sum(weights_sq_subset * phaseMeas[idxSt:idxSt + int(numPhaseMeas[idx])])
        else:
            blockLL[idx, :] = np.sum(Hoverall[idxSt:idxSt + int(numPhaseMeas[idx]), :], axis=0)
            sigOffset[idx] = np.sum(phaseMeas[idxSt:idxSt + int(numPhaseMeas[idx])])

    # the following assumes the phase offset error is added onto each measurement
    Herror = np.zeros((nBrightField, nBrightField))
    sigOffset2use = np.zeros(nBrightField)
    blockLL2use = np.zeros((nBrightField, nZernike))

    # group pairs whose index-wise separation is fixed. Each
    # row contains groups that share the same separation.
    tempIdxblockLL = np.reshape(np.arange(len(useimg)), (nPairs2use_eachSpectrum, -1))  # Index

    # intermediate sum of contributions from each fixed distance group.
    tempSum = np.reshape(weightsSqSum, (nPairs2use_eachSpectrum, -1))

    for idxPair in range(nPairs2use_eachSpectrum):
        vecWeightsSum = tempSum[idxPair, :]
        HerrorEach = np.diag(vecWeightsSum + np.roll(vecWeightsSum, [idxPair+1]))
        HerrorEach -= np.roll(np.diag(vecWeightsSum), [idxPair+1],axis = 1)
        HerrorEach -= np.roll(np.diag(vecWeightsSum), [idxPair+1],axis = 1).T
        Herror += HerrorEach

        blockLLEach = blockLL[tempIdxblockLL[idxPair, :], :]
        blockLLEach -= np.roll(blockLLEach, idxPair+1, axis=0)
        blockLL2use += blockLLEach

        sigOffsetEach = sigOffset[tempIdxblockLL[idxPair, :]]
        sigOffsetEach -= np.roll(sigOffsetEach, idxPair+1, axis=0)
        sigOffset2use += sigOffsetEach

    Hnew = np.block([[Hoverall.T @ Hoverall, blockLL2use.T], [blockLL2use, Herror]])
    H2use = Hnew[:, :-1]     # Due to the fact that a global phase offset added onto
                             # all measurements do not change their subtractions, we
                             # assume that the last measurement has phase offset error equals zero.

    if useWeights:
        sig2use = np.concatenate([Hoverall.T @ (weightsVct * phaseMeas), sigOffset2use])
    else:
        sig2use = np.concatenate([Hoverall.T @ phaseMeas, sigOffset2use])

    zernikeCoeff_temp = np.linalg.solve(H2use.T @ H2use + np.eye(nZernike + nBrightField - 1), H2use.T @ sig2use)
    zernikeCoeff_new = zernikeCoeff_temp[:nZernike]
    ## generate the extracted aberration
    temp = np.reshape(Hz @ zernikeCoeff_new, (2 * maxCTF + 1, 2 * maxCTF + 1)).T  # We should add .T to make it the same as matlab
    CTF_abe = CTF.astype(complex)
    bdC = calBoundary([xc, yc], [2 * maxCTF + 1, 2 * maxCTF + 1])
    CTF_abe[bdC[0,0]-1:bdC[1,0], bdC[0,1]-1:bdC[1,1]] *= np.exp(1j * temp)

    ## Extract the coefficients
    zernikeCoeff = np.zeros(max(zernikeMode2Opt) + 1)
    zernikeCoeff[zernikeMode2Opt] = zernikeCoeff_new

    if marginPix > 0:
        expandFactor = (marginPix + maxCTF) / maxCTF

        fitAbe = cv2.resize(phase_unwrapCG(np.angle(CTF_abe)),
                            (int(ceil(xsize * expandFactor)), int(ceil(ysize * expandFactor))),
                            interpolation=cv2.INTER_LINEAR)
        bd = calBoundary(np.floor(np.array(fitAbe.shape) / 2) + 1, [xsize, ysize])
        fitAbe = fitAbe[int(bd[0,0]-1):int(bd[1,0]), int(bd[0,1]-1):int(bd[1,1])]

        temp = CTF_abe * (np.abs(CTF_abe) > 0.01) + (np.abs(CTF_abe) * (np.abs(CTF_abe) <= 0.01) + 1e-8) * np.exp(
            1j * fitAbe)
        CTF_abe = CTF_abe * (np.abs(CTF_abe) > 0.01) + (np.abs(CTF_abe) * (np.abs(CTF_abe) <= 0.01) + 1e-8) * np.exp(
            1j * medfilt2d(phase_unwrapCG(np.angle(temp)), kernel_size=3))

    return CTF_abe, zernikeCoeff
