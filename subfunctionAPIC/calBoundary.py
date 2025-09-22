import numpy as np

def calBoundary(center, mylength):
    """
    Calculate the boundary for a box (side length L) that centers at point O.
    The first input should be the center, and the second should be the side length of the box.
    When the side length is a scalar, it generates a square.
    """
    # Ensure mylength is a numpy array for consistent processing
    mylength = np.asarray(mylength)

    # If mylength is a scalar, create an array with the same value for each dimension of the center
    if mylength.size == 1:
        mylength = np.full_like(center, mylength)

    # Initialize the boundary array
    bound = np.zeros((2, mylength.size), dtype=int)

    # Calculate the boundary for each dimension
    for ii in range(mylength.size):
        len_c = mylength[ii] // 2 + 1
        bound[0, ii] = center[ii] - len_c + 1
        bound[1, ii] = bound[0, ii] + mylength[ii] - 1

    return bound


