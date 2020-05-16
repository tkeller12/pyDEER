import numpy as np
import os
from scipy.optimize import least_squares, minimize

def kernel(t, r, angles = 5000):
    '''Return the Kernel Matrix.

    .. math::
        K(r,t) = \int_{0}^{\pi/2} \cos(\\theta) \cos[(3 \cos(\\theta)^2 - 1)\omega_{ee} t] d\\theta

        \omega_{ee} = \\frac{\gamma_e^2\hbar}{r^3}

    Args:
        t (numpy.ndarray): Array of time values in seconds
        r (numpy.ndarray): Array of radius (distance) values in meters

    Returns:
        kernel_matrix (numpy.ndarray): Numpy array of kernel. The first dimension is the time dimension. The second dimension is the distance dimension.

    .. note::
        The distance array (r) must have all values greater than zero.

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

def deer_trace(t, r, angles=1000):
    '''Calculate the DEER trace corresponding to a given time axes and distance value

    Args:
        t (numpy.ndarray): Time axes of DEER trace
        r (float,int,numpy.ndarray): Distances value or values in meters
        angles (int): Number of angles to average when generating DEER trace

    Returns:
        trace (numpy.ndarray): DEER trace

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
    theta_array = np.r_[0.:np.pi/2.:1j*angles]

    omega_ee = 2.*np.pi*(5.204e-20)/(r**3.)

    trace = np.zeros_like(t)
    for theta in theta_array:
        omega = (omega_ee)*(2.*(np.cos(theta)**2.)-1.)
        trace = trace + np.cos(theta)*np.cos(omega*t)

    # Normalize by number of angles
    trace = trace / angles
    return trace

def background(t, tau, A, B, d = 3.):
    '''DEER Background function

    .. math::
        A + B e^{- t^{d/3}/\\tau}

    Args:
        t (numpy.ndarray): Time axes for background function
        tau (float): Constant 
        A (float): Offset
        B (float): Scaling factor
        d (float): dimensionality of background function

    Returns:
        background_signal (numpy.ndarray)
    '''
    background_signal = A + B*np.exp(-1*(np.abs(t)**(d/3.))/tau)
    return background_signal

def background_x0(data, t):
    '''Guess initial parameters for background function

    Args:
        data (numpy.ndarray): Array of data
        t (numpy.ndarray): Array of axes

    Returns:
        x0 (list): List of parameters for fit initial guess
    '''

    A = data[-1]
    B = np.max(data) - A
    tau = 1e-6
    d = 3.
    
    x0 = [tau, A, B]
    return x0

def fit_background(data, t, background_function = background, t_min = 0.):
    '''Fit DEER data to background function

    Args:
        data (numpy.ndarray): Array of data values
        t (numpy.ndarray): Array of time axis values
        background_function (func): Background function
        t_min (float): Start time for fit

    Returns:
        fit (numpy.ndarray): Fit of data
    '''

    def res(x, data, t):
        residual = data - background_function(t,*x)
        return residual

    x0 = background_x0(data,t)

    # select range of data for fit
    data_fit = data[t >= t_min]
    t_fit = t[t >= t_min]

    out = least_squares(res,x0,verbose = 2,args = (data_fit,t_fit))
    x = out['x']

    
    fit = background_function(t,*x)

    return fit

def operator(n, L):
    '''Return operator for Regularization

    Args:
        n (int): Number of points in Kernal distance dimension
        L (str, numpy.ndarray): String identifying name of operator or numpy array for operator to pass through function

    Returns:
        L (numpy.ndarray): Regularization operator as numpy array


    +-------------------+----------------------+
    |Operator (L)       |Description           |
    +===================+======================+
    |'Identity'         |Identity Matrix       |
    +-------------------+----------------------+
    |'1st Derivative'   |1st Derivative Matrix |
    +-------------------+----------------------+
    |'2nd Derivative'   |2nd Derivative Matrix |
    +-------------------+----------------------+

    '''
    if L == None or L == 'Identity':
        L = np.eye(n)
    elif L == '1st Derivative':
        L = np.diag(-1.*np.ones(n),k = 0)
        L += np.diag(1.*np.ones(n-1),k = 1)
    elif L == '2nd Derivative':
        L = np.diag(1.*np.ones(n),k = 0)
        L += np.diag(-2.*np.ones(n-1),k = 1)
        L += np.diag(1.*np.ones(n-2),k = 2)

    return L


def tikhonov(K, S, lambda_ = 1.0, L = None):
    '''Perform Tikhonov Regularization

    .. math::
        P_\lambda = {(K^TK + \lambda^2 L^TL)}^{-1} K^TS

    Args:
        K (numpy.ndarray): Kernel Matrix
        S (numpy.ndarray): Experimental DEER trace
        lambda_ (float): Regularization parameter
        L (None, numpy.ndarray): Tikhonov regularization operator, uses identity if argument is None

    Returns: 
        P_lambda (numpy.ndarray): Distance distribution from Tikhonov regularization
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
        L (None, numpy.ndarray): Tikhonov regularization operator, uses identity if argument is None

    Returns:
        (tuple): tuple containing:
        
            rho_array (numpy.ndarray): Residual Norm

            eta_array (numpy.ndarray): Solution Norm

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
        lambda_ (float): 

    Returns:
        P_lambda (numpy.ndarray): Distance distribution minimized by maximum entropy method.
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

def model_free(K, S, lambda_, L = None):
    '''Model Free P(r) with non-negative constraints

    .. math::
        \Phi_{MF}[P] = \|K P(r) - S\|^2 + \lambda^2 \| LP \|^2 \\Rightarrow \min

    Args:
        K (numpy.ndarray): Kernel Matrix
        S (numpy.ndarray): Data array
        lambda_ (float): Regularization parameter
        L (str, numpy.ndarray): Operator for regularization

    Returns:
        P_lambda (numpy.ndarray): Distance distribution
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

#    cons = ({'type':'ineq','fun':lambda x: 0})

    output = minimize(min_func, x0, args = (K, S, lambda_, L), bounds = bounds, options = {'disp':True})

    P_lambda = output['x']

    return P_lambda

    
def gaussian(r, sigma, mu, Normalize = False):
    '''Return Gaussian Distribution from given distance array, standard deviation, and mean distance

    .. math::
        \\frac{1}{\sqrt{2 \pi {\sigma}^2}} e^{-{(r-\mu)}^2/(2\sigma^2)}

    Args:
        r (numpy.ndarray): Numpy array of distance values
        sigma (float): Standard deviation
        mu (float): Mean distance
        Normalize (bool): If True, the integral of Gaussian is normalized to 1

    Returns:
        gaussian_dist (numpy.ndarray): Gaussian distribution
    '''
    if Normalize:
        gaussian_dist = (1./(np.sqrt(2*np.pi*(sigma**2.)))) * np.exp(-1*(r-mu)**2./(2.*(sigma**2.)))
    else:
        gaussian_dist = np.exp(-1*(r-mu)**2./(2.*(sigma**2.)))
    return gaussian_dist

def gaussians(r, x):
    '''Return Gaussian Distribution from given distance array, standard deviation, and mean distance

    .. math::
        \sum_{i = 1}^{N} \\frac{A_i}{\sqrt{2 \pi {\sigma_i}^2}} e^{-{(r-\mu_i)}^2/(2\sigma_i^2)}

    Args:
        r (numpy.ndarray): Numpy array of distance values
        x (list): list of lists. Each gaussian is definied by a list of 3 parameters.

    Returns:
        gaussian_dist (numpy.ndarray): Gaussian distribution
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
        tuple containing:
        
            P_gauss (numpy.ndarray): distance distribution

            x_fit (dict): Dictionary of fitting parameters

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

    if x0 == None:
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
    from matplotlib.pylab import *

    t = np.r_[-0.1e-6:5e-6:500j]
    r = np.r_[1.5e-9:8e-9:100j]

    b = background(t, 10e-6, 10.e3, 10.e3, d = 3.)
    x0 = background_x0(b,t)
    print(x0)
    b_guess = background(t,*x0)

    figure()
    plot(t,b)
    plot(t,b_guess,'r-')

    data = b
    fit = fit_background(data,t)

    figure()
    plot(t,b)
    plot(t,fit,'r--')

    trace = deer_trace(t,4e-9,angles = 100)
    gaussian_dist = gaussian(r,0.5e-9,5e-9)
    kernel_matrix = kernel(t,r,angles = 2000)

    deer = np.dot(kernel_matrix,gaussian_dist)
    deer = deer / np.max(deer)
    noise = 0.01*np.random.randn(len(deer))
    data = deer + noise

    P_calc = tikhonov(kernel_matrix,data,5.,L = '1st Derivative')
    P_calc = P_calc/np.max(P_calc)

    figure()
    plot(r,P_calc)
    show()

    gaussian_dist = gaussian_dist / np.max(gaussian_dist)

    lambda_array = np.logspace(-4,2,100)

    residual_norm, solution_norm = L_curve(kernel_matrix, data, lambda_array)

#    P_max_entropy = maximum_entropy(kernel_matrix,data,20.)
    P_max_entropy = model_free(kernel_matrix,data,5.)

    figure('maximum entropy')
    plot(r,P_max_entropy)

    figure('maximum entropy fit')
    deer = np.dot(kernel_matrix,P_max_entropy)
    plot(t,data)
    plot(t,deer,'r-')


    figure()
    loglog(residual_norm,solution_norm)

    figure()
    plot(t*1e6,trace)
    ylim(-1.1,1.1)
    xlabel('Time (us)')
    figure()
    plot(t*1e6,kernel_matrix)
    ylim(-1.1,1.1)
    xlabel('Time (us)')
    figure()
    plot(r,gaussian_dist)
    figure()
    plot(t*1e6,deer,'b-')
    plot(t*1e6,data,'g-')
    figure()
    plot(r,gaussian_dist,'b-',label = 'exact')
    plot(r,P_calc,'r-',label = 'tik')
    show()

