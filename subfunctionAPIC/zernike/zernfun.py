import numpy as np

def zernfun(n, m, r, theta, nflag=None):
    """
        ZERNFUN Zernike functions of order N and frequency M on the unit circle.
          Z = ZERNFUN(N,M,R,THETA) returns the Zernike functions of order N
          and angular frequency M, evaluated at positions (R,THETA) on the
          unit circle.  N is a vector of positive integers (including 0), and
          M is a vector with the same number of elements as N.  Each element
          k of M must be a positive integer, with possible values M(k) = -N(k)
          to +N(k) in steps of 2.  R is a vector of numbers between 0 and 1,
          and THETA is a vector of angles.  R and THETA must have the same
          length.  The output Z is a matrix with one column for every (N,M)
          pair, and one row for every (R,THETA) pair.

          Z = ZERNFUN(N,M,R,THETA,'norm') returns the normalized Zernike
          functions.  The normalization factor sqrt((2-delta(m,0))*(n+1)/pi),
          with delta(m,0) the Kronecker delta, is chosen so that the integral
          of (r * [Znm(r,theta)]^2) over the unit circle (from r=0 to r=1,
          and theta=0 to theta=2*pi) is unity.  For the non-normalized
          polynomials, max(Znm(r=1,theta))=1 for all [n,m].

          The Zernike functions are an orthogonal basis on the unit circle.
          They are used in disciplines such as astronomy, optics, and
          optometry to describe functions on a circular domain.

          The following table lists the first 15 Zernike functions.

              n    m    Zernike function             Normalization
              ----------------------------------------------------
              0    0    1                              1/sqrt(pi)
              1    1    r * cos(theta)                 2/sqrt(pi)
              1   -1    r * sin(theta)                 2/sqrt(pi)
              2    2    r^2 * cos(2*theta)             sqrt(6/pi)
              2    0    (2*r^2 - 1)                    sqrt(3/pi)
              2   -2    r^2 * sin(2*theta)             sqrt(6/pi)
              3    3    r^3 * cos(3*theta)             sqrt(8/pi)
              3    1    (3*r^3 - 2*r) * cos(theta)     sqrt(8/pi)
              3   -1    (3*r^3 - 2*r) * sin(theta)     sqrt(8/pi)
              3   -3    r^3 * sin(3*theta)             sqrt(8/pi)
              4    4    r^4 * cos(4*theta)             sqrt(10/pi)
              4    2    (4*r^4 - 3*r^2) * cos(2*theta) sqrt(10/pi)
              4    0    6*r^4 - 6*r^2 + 1              sqrt(5/pi)
              4   -2    (4*r^4 - 3*r^2) * sin(2*theta) sqrt(10/pi)
              4   -4    r^4 * sin(4*theta)             sqrt(10/pi)
              ----------------------------------------------------

          Example 1:

              % Display the Zernike function Z(n=5,m=1)
              x = -1:0.01:1;
              [X,Y] = meshgrid(x,x);
              [theta,r] = cart2pol(X,Y);
              idx = r<=1;
              z = nan(size(X));
              z(idx) = zernfun(5,1,r(idx),theta(idx));
              figure
              pcolor(x,x,z), shading interp
              axis square, colorbar
              title('Zernike function Z_5^1(r,\theta)')

          Example 2:

              % Display the first 10 Zernike functions
              x = -1:0.01:1;
              [X,Y] = meshgrid(x,x);
              [theta,r] = cart2pol(X,Y);
              idx = r<=1;
              z = nan(size(X));
              n = [0  1  1  2  2  2  3  3  3  3];
              m = [0 -1  1 -2  0  2 -3 -1  1  3];
              Nplot = [4 10 12 16 18 20 22 24 26 28];
              y = zernfun(n,m,r(idx),theta(idx));
              figure('Units','normalized')
              for k = 1:10
                  z(idx) = y(:,k);
                  subplot(4,7,Nplot(k))
                  pcolor(x,x,z), shading interp
                  set(gca,'XTick',[],'YTick',[])
                  axis square
                  title(['Z_{' num2str(n(k)) '}^{' num2str(m(k)) '}'])
              end

          See also ZERNPOL, ZERNFUN2.

          Paul Fricker 2/28/2012
    """

    # Validate inputs
    if np.ndim(n) != 1 or  np.ndim(m) != 1:
        raise ValueError('N and M must be vectors.')
    if len(n) != len(m):
        raise ValueError('N and M must be the same length.')
    if any(abs(n - m) % 2 != 0):
        raise ValueError('All N and M must differ by multiples of 2 (including 0).')
    if any(m > n):
        raise ValueError('Each M must be less than or equal to its corresponding N.')
    if any((r > 1) | (r < 0)):
        raise ValueError('All R must be between 0 and 1.')
    if np.ndim(r) != 1 or  np.ndim(theta) != 1:
        raise ValueError('R and THETA must be vectors.')
    if len(r) != len(theta):
        raise ValueError('The number of R- and THETA-values must be equal.')

    # Check Normalization
    isnorm = False
    if nflag is not None and isinstance(nflag, str):
        if nflag.lower() == 'norm':
            isnorm = True
        else:
            raise ValueError('Unrecognized normalization flag.')

    ## Compute the Zernike Polynomials

    # Determine the required powers of r
    m_abs = np.abs(m)
    rpowers = []

    # Calculate the required powers of r
    for j in range(len(n)):
        rpowers.extend(range(m_abs[j], n[j] + 1, 2))

    rpowers = np.unique(rpowers)

    # Pre-compute the values of r raised to the required powers
    if rpowers[0] == 0:
        rpowern = np.array([r ** p for p in rpowers[1:]]).T
        rpowern = np.hstack([np.ones((len(r), 1)), rpowern])
    else:
        rpowern = np.array([r ** p for p in rpowers]).T


    # Compute the values of the polynomials
    z = np.zeros((len(r), len(n)))

    for j in range(len(n)):
        s = np.arange(0, (n[j] - m_abs[j]) // 2 + 1)
        pows = np.arange(n[j], m_abs[j] - 1, -2)

        for k in range(len(s) - 1, -1, -1):
            p = ((1 - 2 * (s[k] % 2)) *
                 np.prod(np.arange(2, n[j] - s[k] + 1)) /
                 np.prod(np.arange(2, s[k] + 1)) /
                 np.prod(np.arange(2, (n[j] - m_abs[j]) // 2 - s[k] + 1)) /
                 np.prod(np.arange(2, (n[j] + m_abs[j]) // 2 - s[k] + 1)))

            idx = pows[k] == rpowers
            z[:, j] += p * rpowern[:, idx].flatten()

        if isnorm:
            z[:, j] *= np.sqrt((1 + (m[j] != 0)) * (n[j] + 1) / np.pi)

    # Compute the Zernike Polynomials
    idx_pos = m > 0
    idx_neg = m < 0

    # Compute the Zernike functions
    if np.any(idx_pos):
        z[:, idx_pos] = z[:, idx_pos] * np.cos(theta[:, None] * m_abs[idx_pos])
    if np.any(idx_neg):
        z[:, idx_neg] = z[:, idx_neg] * np.sin(theta[:, None] * m_abs[idx_neg])


    return z

# Define zernike_poly function here based on the mathematical definition of Zernike polynomials
