import numpy as np
from subfunctionAPIC.calCoord import calCoord
from subfunctionAPIC.calBoundary import calBoundary
def F(x):
    return np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(x)))

def iF(x):
    return np.fft.ifftshift(np.fft.ifft2(np.fft.fftshift(x)))

# 定义对数幅度函数
def logamp(x):
    return np.log10(np.abs(x) + 1)


def preprocessing(NA_seq_BF, NA_cell, imlow,enableROI,ROILength,ROIcenter,lambda_g, dpix_c, mag, NA):
    """
    Perform preprocessing
    """
    xsize, ysize = imlow.shape[:2]
    Nled = imlow.shape[2]

    ROI_c = [np.ceil((xsize + 1) / 2), np.ceil((ysize + 1) / 2)]
    if enableROI:
        if isinstance(ROIcenter, (list, tuple, np.ndarray)) and len(ROIcenter) == 2:
            bdROI = calBoundary(ROIcenter, ROILength)
        elif isinstance(ROIcenter, str) and ROIcenter.lower() == 'auto':
            bdROI = calBoundary(ROI_c, ROILength)
        else:
            print("ROI exceeds the boundary. Set to be 'auto' ")
            bdROI = calBoundary(ROI_c, ROILength)
    else:
        bdROI = np.array([1, 1, xsize, ysize])

    I_low = imlow[bdROI[0, 0] - 1:bdROI[1, 0], bdROI[0, 1] - 1:bdROI[1, 1], :]

    ## Calculate NA
    NAx_vis_BF = -NA_seq_BF[:, 1]
    NAy_vis_BF = -NA_seq_BF[:, 0]
    # Plot the calibration results

    na_design_BF = np.array([NAx_vis_BF, NAy_vis_BF]).T
    # Calling the calCoord function with the specified parameters
    freqXY_BF, con, _, _, _, _, _, _ = calCoord(na_design_BF / lambda_g, [ROILength, ROILength], dpix_c, mag, NA,
                                             lambda_g)

    # Recalibrating na_rp_cal and freqXY_calib based on the outputs
    na_rp_cal = NA / lambda_g * con
    freqXY_calib_BF = freqXY_BF
    na_calib_BF = na_design_BF

    freqXY_calib_DF = []
    na_calib_DF = []
    for DF_index in range(len(NA_cell)):
        NAx_vis_DF = -NA_cell[DF_index][:, 1]
        NAy_vis_DF = -NA_cell[DF_index][:, 0]
        # Plot the calibration results


        na_design_DF = np.array([NAx_vis_DF, NAy_vis_DF]).T
        # Calling the calCoord function with the specified parameters
        freqXY_DF_temp, con, _, _, _, _, _, _ = calCoord(na_design_DF / lambda_g, [ROILength, ROILength], dpix_c, mag, NA,
                                                 lambda_g)

        # Recalibrating na_rp_cal and freqXY_calib based on the outputs
        freqXY_calib_DF.append(freqXY_DF_temp)
        na_calib_DF.append(na_design_DF)

    return I_low, na_calib_BF,na_calib_DF, freqXY_calib_BF,freqXY_calib_DF,na_rp_cal,bdROI




