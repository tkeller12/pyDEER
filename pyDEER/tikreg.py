import numpy as np
import os
from scipy.optimize import least_squares, minimize
from scipy.special import fresnel

def autophase(S):
    '''Optimize phase of complex data by maximizing the sum of imaginary over sum of real

    .. math::
        \phi = \\arctan \left( \\frac{\sum_i^N \Im(s_i) }{  \sum_i^N \Re(s_i) } \\right)

        S_{\phi} = S e^{-i \phi}

    Args:
        S (numpy.ndarray): Complex data

    Returns:
        numpy.ndarray: Automatically phased complex data
    '''

    phase = np.arctan(np.sum(np.imag(S))/np.sum(np.real(S)))

    S_phased = np.exp(-1j * phase) * S

    return S_phased

def add_noise(S,sigma):
    '''Add noise to array
    
    Args:
        S (numpy.ndarray): Array to add noise to
        sigma (float): Standard deviation of noise
        
    Returns:
        numpy.ndarray: Array with noise added
    '''

    S_noisy = S + sigma * np.random.randn(*np.shape(S))

    return S_noisy

def kernel(t, r, method = 'fresnel', angles = 5000):
    '''Return the Kernel Matrix.

    .. math::
        K(r,t) = \int_{0}^{\pi/2} \cos(\\theta) \cos[(3 \cos(\\theta)^2 - 1)\omega_{ee} t] d\\theta

        \omega_{ee} = \\frac{\gamma_e^2\hbar}{r^3}

    +-------------------+----------------------+
    |Method             |Description           |
    +===================+======================+
    |'fresnel'          |Fresnel Integral      |
    +-------------------+----------------------+
    |'brute force'      |Brute Force Method    |
    +-------------------+----------------------+

    Args:
        t (numpy.ndarray): Array of time values in seconds
        r (numpy.ndarray): Array of radius (distance) values in meters
        method (str): Method for calculating the kernel. By default, uses the fresnel integral
        angles (int): For brute-force kernel, number of angles to average over

    Returns:
        numpy.ndarray: Numpy array of kernel. The first dimension is the time dimension. The second dimension is the distance dimension.

    .. note::
        The distance array (r) must have all values greater than zero to generate a proper kernel.

    .. warning::
        The number of angles must be carefully selected to ensure the Kernel matrix properly averages the angles for short distances.

    Example::
    
        t = np.r_[-0.1e-6:10e-6:1000j]
        r = np.r_[1.5e-9:10e-9:100j]
        K = kernel(t,r,angles = 2000)
    '''

    t = t.reshape(-1,1)
    r = r.reshape(1,-1)
    K = deer_trace(t,r,angles=angles)

    return K

def load_kernel(filename = 'default_kernel.csv', directory = 'kernels'):
    '''Import Kernel Matrix
    '''
    full_path = os.path.join(directory, filename)
    kernel_matrix = np.loadtxt(full_path,delimiter = ',')
    return kernel_matrix

def save_kernel(k, filename, directory = 'kernels'):
    '''Save Kernel Matrix

    Args:
        filename (str): Kernel filename
        k (numpy.ndarray): Kernel Matrix
        directory (str): Path to Kernel filename
    '''
    full_path = os.path.join(directory,filename)

    np.savetxt(full_path,k,delimiter = ',')

def background_dist(t):
    '''Calculate the distance above which P(r) should be zero in background fit.

    Args:
        t (numpy.ndarray): Time axes

    Returns:
        r (float): Distance value for background fit

    '''

    oscillations = 2.
    omega_ee = 2.*np.pi * oscillations / np.max(t)

    r = ((2. * np.pi * 5.204e-20)/omega_ee)**(1./3.)

    return r


def deer_trace(t, r, method = 'fresnel', angles=1000):
    '''Calculate the DEER trace corresponding to a given time axes and distance value

    +-------------------+----------------------+
    |Method             |Description           |
    +===================+======================+
    |'fresnel'          |Fresnel Integral      |
    +-------------------+----------------------+
    |'brute force'      |Brute Force Method    |
    +-------------------+----------------------+

    Args:
        t (numpy.ndarray): Time axes of DEER trace
        r (float, int, numpy.ndarray): Distances value or values in meters
        method (str): Method for calculating deer trace, by default uses fresnel integral
        angles (int): For brute force method, number of angles to average when generating DEER trace

    Returns:
        numpy.ndarray: DEER trace

    Example::

        import numpy as np
        from matplotlib.pylab import *

        r = 4e-9
        t = np.r[0.:10e-6:1000j]
        trace = deer_trace(t,r)

        figure()
        plot(t,trace)
        show()
    '''

    omega_ee = 2.*np.pi*(5.204e-20)/(r**3.)

    if method == 'brute force':
        theta_array = np.r_[0.:np.pi/2.:1j*angles]
        trace = np.zeros_like(t)
        for theta in theta_array:
            omega = (omega_ee)*(3.*(np.cos(theta)**2.)-1.)
            trace = trace + np.sin(theta)*np.cos(omega*t)

        # Normalize by number of angles and Fresnel Integral
        trace = trace / (angles * (np.sqrt(np.pi/8.)))
    elif method == 'fresnel':
        x = np.sqrt(6.*omega_ee*np.abs(t))/ np.sqrt(np.pi)

        S, C = fresnel(x)

        trace = np.cos(omega_ee*t)*(C/x) + np.sin(omega_ee*np.abs(t))*(S/x)

    return trace

def background(t, tau, A, B, d = 3.):
    '''DEER Background function

    .. math::
        A + B e^{- t^{d/3}/\\tau}

    Args:
        t (numpy.ndarray): Time axes for background function
        tau (float): Time constant 
        A (float): Offset
        B (float): Scaling factor
        d (float): dimensionality of background function

    Returns:
        numpy.ndarray: Background signal
    '''
    background_signal = A + B*np.exp(-1*(np.abs(t)**(d/3.))/tau)
    return background_signal

def background_x0(t, data):
    '''Guess initial parameters for background function

    Args:
        data (numpy.ndarray): Array of data
        t (numpy.ndarray): Array of axes

    Returns:
        list: List of parameters for fit initial guess
    '''

    A = data[-1]
    B = np.max(data) - A
    tau = 10e-6
    d = 3.
    
    x0 = [tau, A, B]
    return x0

def tikhonov_background(t, r, K, data, background_function = background, r_background = None, lambda_ = 1., L = 'Identity', x0 = None):
    '''Fit DEER data to background function by forcing P(r) to be zero 

    Args:
        t (numpy.ndarray): Array of time axis values
        r (numpy.ndarray): Array of distance values for Kernel
        K (numpy.ndarray): Kernel matrix
        data (numpy.ndarray): Array of data values
        background_function (func): Background function
        r_background (float): Distance above which P(r) is optimized to zero
        lambda_ (float): Regularization parameter
        L (str, numpy.ndarray): Regularization operator, by default Identity for background optimization
        x0 (list): Initial guess for background fit parameters

    Returns:
        numpy.ndarray: Background fit of data
    '''

    # If None, determine r_background based on time trace
    if r_background == None:
        r_background = background_dist(t)

    # If None, initial guess for background function
    if x0 is None:
        x0 = background_x0(t, data)

    def res(x, data, t, r, K, r_background):
        P_tik = tikhonov(K, (data / background_function(t, *x)) - 1., lambda_ = lambda_, L = L)

        P_tik[r < r_background] = 0.
        residual = P_tik

        return residual

    out = least_squares(res, x0, verbose = 2, args = (data, t, r, K, r_background), method = 'lm')

    x = out['x']

    fit = background_function(t, *x)

    return fit

def exp_background(t, data, background_function = background, t_min = 0., x0 = None):
    '''Fit DEER data to background function

    Args:
        t (numpy.ndarray): Array of time axis values
        data (numpy.ndarray): Array of data values
        background_function (func): Background function
        t_min (float): Start time for fit
        x0 (list): Initial guess for background fit parameters

    Returns:
        numpy.ndarray: Fit of data
    '''

    if x0 == None:
        x0 = background_x0(t, data)

    def res(x, t, data):
        residual = data - background_function(t, *x)
        return residual

    # select range of data for fit
    data_fit = data[t >= t_min]
    t_fit = t[t >= t_min]

    out = least_squares(res, x0, verbose = 2, args = (t_fit, data_fit), method = 'lm')
    x = out['x']

    fit = background_function(t,*x)

    return fit

def operator(n, L):
    '''Return operator for Regularization

    +-------------------+----------------------+
    |Operator (L)       |Description           |
    +===================+======================+
    |'Identity'         |Identity Matrix       |
    +-------------------+----------------------+
    |'1st Derivative'   |1st Derivative Matrix |
    +-------------------+----------------------+
    |'2nd Derivative'   |2nd Derivative Matrix |
    +-------------------+----------------------+

    Args:
        n (int): Number of points in Kernal distance dimension
        L (str, numpy.ndarray): String identifying name of operator or numpy array for operator to pass through function

    Returns:
        numpy.ndarray: Regularization operator as numpy array
    '''
    if L == 'Identity':
        L = np.eye(n)
    elif L == '1st Derivative':
        L = np.diag(-1.*np.ones(n),k = 0)
        L += np.diag(1.*np.ones(n-1),k = 1)
        L = L[:-1,:]
    elif (L == None) or (L == '2nd Derivative'):
        L = np.diag(1.*np.ones(n),k = 0)
        L += np.diag(-2.*np.ones(n-1),k = 1)
        L += np.diag(1.*np.ones(n-2),k = 2)
        L = L[:-2,:]
    elif isinstance(L, str):
        raise ValueError('Operator string not understood')

    return L


def tikhonov(K, S, lambda_ = 1.0, L = None):
    '''Perform Tikhonov Regularization

    .. math::
        P_\lambda = {(K^TK + \lambda^2 L^TL)}^{-1} K^TS

    Args:
        K (numpy.ndarray): Kernel Matrix
        S (numpy.ndarray): Experimental DEER trace
        lambda_ (float): Regularization parameter
        L (None, numpy.ndarray): Tikhonov regularization operator, uses 2nd derivative if argument is None

    Returns: 
        numpy.ndarray: Distance distribution from Tikhonov regularization
    '''
    # Select Real Part
    S = np.real(S)

    # Set Operator for Tikhonov Regularization
    n = np.shape(K)[1]

    # Determine Operator for Regularization
    L = operator(n,L)

    P_lambda = np.dot(np.linalg.inv(np.dot(K.T, K)+(lambda_**2.)*np.dot(L.T, L)),np.dot(K.T, S))

    return P_lambda

def L_curve(K, S, lambda_array, L = None):
    '''Generate Tikhonov L-curve 

    Args:
        K (numpy.ndarray): Kernel Matrix
        S (numpy.ndarray): Experimental DEER trace
        lambda_ (numpy.ndarray): Array of Regularization parameters
        L (None, numpy.ndarray): Tikhonov regularization operator, uses 2nd derivative if argument is None

    Returns:
        tuple: tuple containing:
        
            rho_array (*numpy.ndarray*): Residual Norm

            eta_array (*numpy.ndarray*): Solution Norm

    '''

    rho_list = []
    eta_list = []
    for lambda_ in lambda_array:
        P_lambda = tikhonov(K, S, lambda_, L = L)
        rho_list.append(np.linalg.norm(S - np.dot(K, P_lambda)))
        eta_list.append(np.linalg.norm(P_lambda))
    
    rho_array = np.array(rho_list)
    eta_array = np.array(eta_list)

    return rho_array, eta_array

def maximum_entropy(K, S, lambda_):
    '''Maximum Entropy method for determining P(r)

    .. math::
        \Phi_{ME}[P] = \|K P(r) - S\|^2 + \lambda^2 \\times \int [P(r) \ln \\frac{P(r)}{P_0(r)} + \\frac{P_0(r)}{e}] dr \\Rightarrow \min

    Args:
        K (numpy.ndarray): Kernel Matrix
        S (numpy.ndarray): Data array
        lambda_ (float): Regularization parameter

    Returns:
        numpy.ndarray: Distance distribution minimized by maximum entropy method.
    '''

    def min_func(P, K, S, lambda_):
        res = np.linalg.norm(np.dot(K, P) - S)**2. + (lambda_**2.)*np.sum((P*np.log((P/x0)) + x0/np.exp(1)))
        return res

    x0 = tikhonov(K, S, lambda_)
    x0[x0<=0.] = 1.e-5

    n = np.shape(K)[1]

    bounds = tuple(zip(1e-15*np.ones(n),np.inf*np.ones(n)))

    output = minimize(min_func, x0, args = (K, S, lambda_), method = 'SLSQP', bounds = bounds, options = {'disp':True})

    P_lambda = output['x']

    return P_lambda

def model_free(K, S, lambda_ = 1., L = None):
    '''Model Free P(r) with non-negative constraints

    .. math::
        \Phi_{MF}[P(r)] = \|K P(r) - S\|^2 + \lambda^2 \| LP(r) \|^2 \\Rightarrow \min

    Args:
        K (numpy.ndarray): Kernel Matrix
        S (numpy.ndarray): Data array
        lambda_ (float): Regularization parameter
        L (str, numpy.ndarray): Operator for regularization

    Returns:
        numpy.ndarray: Distance distribution from model free fit
    '''

    def min_func(P, K, S, lambda_, L):
        res = np.linalg.norm(np.dot(K, P) - S)**2. + (lambda_**2.) * (np.linalg.norm(np.dot(L, P))**2.)
        return res

    n = np.shape(K)[1]

    # Determine Operator for Regularization
    L = operator(n,L)

    x0 = tikhonov(K, S, lambda_)
    x0[x0<=0.] = 1.e-5

    bounds = tuple(zip(np.zeros(len(x0)), np.inf*np.ones(len(x0))))

    output = minimize(min_func, x0, args = (K, S, lambda_, L), bounds = bounds, options = {'disp':True})

    P_lambda = output['x']

    return P_lambda

    
def gaussian(r, sigma, mu, Normalize = False):
    '''Return Gaussian Distribution from given distance array, standard deviation, and mean distance

    If Normalize = True:

    .. math::
        \\frac{1}{\sqrt{2 \pi {\sigma}^2}} e^{-{(r-\mu)}^2/(2\sigma^2)}

    If Normalize = False:

    .. math::
        e^{-{(r-\mu)}^2/(2\sigma^2)}

    Args:
        r (numpy.ndarray): Numpy array of distance values
        sigma (float): Standard deviation
        mu (float): Mean distance
        Normalize (bool): If True, the integral of Gaussian is normalized to 1

    Returns:
        numpy.ndarray: Gaussian distribution
    '''
    if Normalize:
        gaussian_dist = (1./(np.sqrt(2*np.pi*(sigma**2.)))) * np.exp(-1*(r-mu)**2./(2.*(sigma**2.)))
    else:
        gaussian_dist = np.exp(-1*(r-mu)**2./(2.*(sigma**2.)))
    return gaussian_dist

def gaussians(r, x):
    '''Return sum of Gaussian distributions from given distance array and list of lists defining amplitude, standard deviation, and mean distance for each Gaussian

    .. math::
        \sum_{i = 1}^{N} A_i e^{-{(r-\mu_i)}^2/(2\sigma_i^2)}

    Args:
        r (numpy.ndarray): Numpy array of distance values
        x (list): list of lists. Each gaussian is definied by a list of 3 parameters. The parameters are ordered: A - amplitude, sigma - standard deviation, mu - center of distribution.

    Returns:
        numpy.ndarray: Gaussian distribution
    '''

    gaussian_dist = np.zeros(len(r))
    for gaussian_parameters in x:
        A = gaussian_parameters[0]
        sigma = gaussian_parameters[1]
        mu = gaussian_parameters[2]

        gaussian_dist += (A * gaussian(r, sigma, mu))

    return gaussian_dist

def model_gaussian(K, S, r, x0 = None):
    '''Gaussian based fit for distance distribution

    Args:
        K (numpy.ndarray): Kernel Matrix
        S (numpy.ndarray): Data array
        r (numpy.ndarray): Array of distance values
        x0 (None, List): Initial guess. If None, the initial guess is automatically chosen based on Tikhonov regularization P(r)
        
    Returns:
        tuple: tuple containing:
        
            P_gauss (*numpy.ndarray*): distance distribution

            x_fit (*dict*): Dictionary of fitting parameters

    '''

    def min_func(x, K, S, r):

        A = x[0]
        sigma = x[1]
        mu = x[2]

        P_fit = A*gaussian(r,sigma,mu)
        S_fit = np.dot(K,P_fit)

        res = sum((S_fit - S)**2.)
        return res

    bounds = tuple(zip(np.zeros(3), np.inf*np.ones(3)))

    if x0 == None: # Find initial guess based on Tikhonov Regularization
        P_lambda = tikhonov(K, S, lambda_ = 1.0, L = None)
        A_0 = np.max(P_lambda) # Amplitude is maximum value
        sigma_0 = 0.2e-9 # Sigma is this value
        mu_0 = r[np.argmax(P_lambda)] # center is maximum
        def guess_min_func(x, P_lambda, r):
            A = x[0]
            sigma = x[1]
            mu = x[2]

            res = sum((A * gaussian(r,sigma,mu) - P_lambda)**2.)
            return res
        x0 = [A_0, sigma_0, mu_0]
        guess_output = minimize(guess_min_func, x0, args = (P_lambda, r),method = 'Nelder-Mead', bounds = bounds, options = {'disp':True})

        A_0 = guess_output['x'][0]
        sigma_0 = guess_output['x'][1]
        mu_0 = guess_output['x'][2]

        x0 = [A_0,sigma_0,mu_0]

#    output = minimize(min_func, x0, args = (K, S, r), bounds = bounds, options = {'disp':True})
    output = minimize(min_func, x0, args = (K, S, r), method = 'Nelder-Mead', options = {'disp':True})

    A = output['x'][0]
    sigma = output['x'][1]
    mu = output['x'][2]
    P_gauss = A * gaussian(r, sigma, mu)

    x_fit = {}
    x_fit['A'] = A
    x_fit['sigma'] = sigma
    x_fit['mu'] = mu

    return P_gauss, x_fit

if __name__ == '__main__':
    pass
