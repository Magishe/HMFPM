import numpy as np
from scipy.fftpack import dct, idct


def phase_unwrapCG(wrappedPhase, weights=None):
    """
    Performs 2D phase unwrapping using conjugate gradient. CPU version

    This implementation is based on the work by Ghiglia and Romero (JOSA A, 1994)
    DOI: 10.1364/JOSAA.11.000107

    Args:
        wrappedPhase (ndarray): The phase to be unwrapped.
        weights (ndarray): Weights to be used in the unwrapping. Ideally, small weights
                           are assigned to places where the phase measurements might be corrupted.

    Returns:
        ndarray: The unwrapped phase.
    """

    # Default weights to ones if not provided
    if weights is None:
        weights = np.ones_like(wrappedPhase)

    # Normalize the weights
    weights = weights / np.max(weights)
    sumW = np.sum(weights)
    if np.any(weights < 0):
        raise ValueError('Weights cannot be negative. Please check your input.')

    # Preparing for calculating the phase
    xsize, ysize = wrappedPhase.shape
    Y, X = np.meshgrid(range(1, ysize + 1), range(1, xsize + 1))
    denominator2use = 2 * (np.cos(np.pi * (X - 1) / xsize) + np.cos(np.pi * (Y - 1) / ysize) - 2)
    denominator2use[0,0] = 1 # Meaningless value
    # Prepare for the modified discrete Laplacian operator
    squaredW = weights ** 2
    weightsLapX = np.minimum(squaredW[1:], squaredW[:-1])
    weightsLapX = np.vstack([weightsLapX, np.zeros((1, squaredW.shape[1]))])

    weightsLapY = np.minimum(squaredW[:, 1:], squaredW[:, :-1])
    weightsLapY = np.hstack([weightsLapY, np.zeros((squaredW.shape[0], 1))])
    dx, dy = dGrad(wrappedPhase)
    dx = wrapToPi(dx)
    dy = wrapToPi(dy)

    unwrappedPhase = np.zeros_like(wrappedPhase)
    r = dLap(dx, dy, weightsLapX, weightsLapY)
    n0 = np.linalg.norm(r)
    threshold = 1e-8
    # Clear the variable 'p' if it exists
    try:
        del p
    except NameError:
        pass  # p was not defined

    while np.any(r != 0):
        dctR = dct2(r)
        temp = dctR / denominator2use
        temp[0, 0] = dctR[0, 0]  # Use the same bias
        z = idct2(temp)

        if 'p' not in locals():
            p = z
            rzNow = np.sum(r * z)
        else:
            rzNow = np.sum(r * z)
            beta = rzNow / rzPrev
            p = z + beta * p
        rzPrev = rzNow

        dx, dy = dGrad(p)
        dx = wrapToPi(dx)
        dy = wrapToPi(dy)
        Qpk = dLap(dx, dy, weightsLapX, weightsLapY)
        alpha = rzNow / np.sum(p * Qpk)
        unwrappedPhase = unwrappedPhase + alpha * p
        r = r - alpha * Qpk

        if np.linalg.norm(r) < threshold * n0 or sumW == np.size(wrappedPhase):
            break


    return unwrappedPhase


def dGrad(X):
    """
    Calculate the discrete gradient of a matrix.

    Args:
        X (ndarray): A 2D numpy array.

    Returns:
        tuple of ndarray: The discrete gradients along the x-axis (dx) and y-axis (dy).
    """
    dx = X[1:, :] - X[:-1, :]
    dx = np.vstack([dx, np.zeros((1, X.shape[1]))])

    dy = X[:, 1:] - X[:, :-1]
    dy = np.hstack([dy, np.zeros((X.shape[0], 1))])

    return dx, dy


def wrapToPi(angles):
    """
    Wrap angle differences to the range [-pi, pi].

    Args:
        angles (ndarray): Array of angles.

    Returns:
        ndarray: Array of wrapped angles.
    """
    return ((angles + np.pi) % (2 * np.pi)) - np.pi


def dLap(dx, dy, wx=None, wy=None):
    """
    Calculate the discrete Laplacian of a matrix.

    Args:
        dx (ndarray): Gradient along the x-axis.
        dy (ndarray): Gradient along the y-axis.
        wx (ndarray, optional): Weights for x-gradient. Defaults to None.
        wy (ndarray, optional): Weights for y-gradient. Defaults to None.

    Returns:
        ndarray: The discrete Laplacian of the matrix.
    """
    L = np.zeros_like(dx)

    if wx is not None and wy is not None:
        dx = wx * dx
        dy = wy * dy

    Lx = dx[1:, :] - dx[:-1, :]
    Ly = dy[:, 1:] - dy[:, :-1]

    L[1:, :] = Lx
    L[:, 1:] += Ly

    # Handling the boundary conditions
    L[0, :] += dx[0, :]
    L[:, 0] += dy[:, 0]

    return L


# Function for 2D Discrete Cosine Transform
def dct2(a):
    return dct(dct(a.T, norm='ortho').T, norm='ortho')


# Function for 2D Inverse Discrete Cosine Transform
def idct2(a):
    return idct(idct(a.T, norm='ortho').T, norm='ortho')
