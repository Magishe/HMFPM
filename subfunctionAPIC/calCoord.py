import numpy as np

def calCoord(freqUV, imSz, dpix_c, mag, NA, lambda_val):
    """
    Convert from k-space coordinates to pixel coordinates
    """

    con = imSz[0] * dpix_c / mag  # Conversion factor (pixels/(1/um))

    # k-space (u, v) coordinates (1/um)
    uCent = freqUV[:, 0]
    vCent = freqUV[:, 1]

    # Real space (x, y) coordinates (pixels)
    xMid = np.floor(imSz[1] / 2) + 1
    yMid = np.floor(imSz[0] / 2) + 1

    # Grid
    xI, yI = np.meshgrid(range(1, imSz[1] + 1), range(1, imSz[0] + 1))

    # k-space coordinates in terms of pixel values
    xCent = xMid + uCent * con
    yCent = yMid + vCent * con

    # Combine into variables for ease of transport
    freqXY = np.vstack((xCent, yCent)).T
    XYmid = [xMid, yMid]

    # Predicted radius (pixels)
    radP = (NA / lambda_val) * con

    # k-space (u, v) coordinates (1/um)
    uI = (xI - xMid) / con
    vI = (yI - yMid) / con

    return freqXY, con, radP, xI, yI, uI, vI, XYmid
