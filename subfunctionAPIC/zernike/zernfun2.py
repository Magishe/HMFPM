import numpy as np
from subfunctionAPIC.zernike.zernfun import zernfun

def zernfun2(p, r, theta, nflag=None):
    """
        %ZERNFUN2 Single-index Zernike functions on the unit circle.
        %   Z = ZERNFUN2(P,R,THETA) returns the Pth Zernike functions evaluated
        %   at positions (R,THETA) on the unit circle.  P is a vector of positive
        %   integers between 0 and 35, R is a vector of numbers between 0 and 1,
        %   and THETA is a vector of angles.  R and THETA must have the same
        %   length.  The output Z is a matrix with one column for every P-value,
        %   and one row for every (R,THETA) pair.
        %
        %   Z = ZERNFUN2(P,R,THETA,'norm') returns the normalized Zernike
        %   functions, defined such that the integral of (r * [Zp(r,theta)]^2)
        %   over the unit circle (from r=0 to r=1, and theta=0 to theta=2*pi)
        %   is unity.  For the non-normalized polynomials, max(Zp(r=1,theta))=1
        %   for all p.
        %
        %   NOTE: ZERNFUN2 returns the same output as ZERNFUN, for the first 36
        %   Zernike functions (order N<=7).  In some disciplines it is
        %   traditional to label the first 36 functions using a single mode
        %   number P instead of separate numbers for the order N and azimuthal
        %   frequency M.
        %
        %   Example:
        %
        %       % Display the first 16 Zernike functions
        %       x = -1:0.01:1;
        %       [X,Y] = meshgrid(x,x);
        %       [theta,r] = cart2pol(X,Y);
        %       idx = r<=1;
        %       p = 0:15;
        %       z = nan(size(X));
        %       y = zernfun2(p,r(idx),theta(idx));
        %       figure('Units','normalized')
        %       for k = 1:length(p)
        %           z(idx) = y(:,k);
        %           subplot(4,4,k)
        %           pcolor(x,x,z), shading interp
        %           set(gca,'XTick',[],'YTick',[])
        %           axis square
        %           title(['Z_{' num2str(p(k)) '}'])
        %       end
        %
        %   See also ZERNPOL, ZERNFUN.

        %   Paul Fricker 11/13/2006
        %   Implemented in Python Shi Zhao 11/18/2023
    """

    # Check and prepare the inputs:
    if np.ndim(p) != 1:
        raise ValueError('Input p must be a vector.')

    if np.any(p) > 35:
        raise ValueError('zernfun2 only computes the first 36 Zernike functions (p = 0 to 35).')

    # Get the order and frequency corresonding to the function number:
    p = np.array(p).flatten()
    n = np.ceil((-3 + np.sqrt(9 + 8 * p)) / 2).astype(int)
    m = 2 * p - n * (n + 2)

    # Call zernfun (you need to have this function implemented)
    if nflag is None:
        z = zernfun(n, m, r, theta)
    else:
        z = zernfun(n, m, r, theta, nflag)

    return z


