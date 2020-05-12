import numpy as np
import os
from scipy.optimize import least_squares, minimize

def kernel(t,r,angles = 5000):
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

def load_kernel(filename = 'default_kernel.csv',directory = 'kernels'):
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

def deer_trace(t,r,angles=1000):
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

def background(t,tau, A, B, d = 3.):
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

def background_x0(data,t):
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

def fit_background(data,t,background_function = background,t_min = 0.):
    '''Fit DEER data to background function

    Args:
        data (numpy.ndarray): Array of data values
        t (numpy.ndarray): Array of time axis values
        background_function (func): Background function
        t_min (float): Start time for fit

    Returns:
        fit (numpy.ndarray): Fit of data
    '''

    def res(x,data,t):
        '''Calculate Residual

        Args:
            x (list): List of fitting parameters
            data (numpy.ndarray): Array of data values
            t (numpy.ndarray): Array of time values
        '''

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


def tikhonov(K, S, lambda_ = 1.0, L = None):
    '''Perform Tikhonov Regularization

    .. math::
        P_\lambda = {(K^TK + \lambda L^TL)}^{-1} K^TS

    Args:
        K (numpy.ndarray): Kernel Matrix
        S (numpy.ndarray): Experimental DEER trace
        lambda_ (float): Regularization parameter
        L (None, numpy.ndarray): Tikhonov regularization operator, uses identity if argument is None

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
    # Select Real Part
    S = np.real(S)

    # Set Operator for Tikhonov Regularization
    n = np.shape(K)[1]
    if L == None or L == 'Identity':
        L = np.eye(n)
    elif L == '1st Derivative':
        L = np.diag(-1.*np.ones(n),k = 0)
        L += np.diag(1.*np.ones(n-1),k = 1)

    elif L == '2nd Derivative':
        n = np.shape(K)[1]
#        L = np.zeros((n,n))
        L = np.diag(1.*np.ones(n),k = 0)
        L += np.diag(-2.*np.ones(n-1),k = 1)
        L += np.diag(1.*np.ones(n-2),k = 2)

#        L = L[:,0:n-2]
#        L = L[0:n-2,:]
#        print(L)


    P_lambda = np.dot(np.linalg.inv(np.dot(K.T, K)+(lambda_**2.)*np.dot(L.T, L)),np.dot(K.T, S))

    return P_lambda

def L_curve(K, S, lambda_array, L = None):
    '''Generate Tikhonov L-curve 

    Args:
        K (numpy.ndarray): Kernel Matrix
        S (numpy.ndarray): Experimental DEER trace
        lambda_ (numpy.ndarray): Array of Regularization parameters
        operator (None, numpy.ndarray): Tikhonov regularization operator, uses identity if argument is None

    Returns:
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

    Args:
        K (numpy.ndarray): Kernel Matrix
        S (numpy.ndarray): Data array
        lambda_ (float): 
    '''

    def min_func(P, K, S, lambda_):
        res = np.linalg.norm(np.dot(K, P) - S)**2. + (lambda_**2.)*np.sum((P*np.log(P)))
        print(res)
        return res

    x0 = tikhonov(K, S, lambda_)
    x0[x0<=0.] = 1.e-3
#    x0 = np.ones(len(x0))

    print(x0)
    bounds = tuple(zip(np.zeros(len(x0)),100.*np.ones(len(x0))))
    print(bounds)

    cons = ({'type':'ineq','fun':lambda x: 0})

    output = minimize(min_func,x0,args = (K, S, lambda_),method = 'Nelder-Mead',bounds = bounds,constraints = cons)

    P_lambda = output['x']

    return P_lambda

def model_free(K, S, lambda_, L = None):
    '''Maximum Entropy method for determining P(r)

    Args:
        K (numpy.ndarray): Kernel Matrix
        S (numpy.ndarray): Data array
        lambda_ (float): 
        L (func): operator
    '''

    def min_func(P, K, S, lambda_, L):
        res = np.linalg.norm(np.dot(K, P) - S)**2. + (lambda_**2.) * (np.linalg.norm(np.dot(L, P))**2.)
        print(res)
        return res

    if L == None:
        L = np.eye(np.shape(K)[1])

    x0 = tikhonov(K, S, lambda_)
    x0[x0<=0.] = 1.e-3
#    x0 = np.ones(len(x0))

    print(x0)
    bounds = tuple(zip(np.zeros(len(x0)),100.*np.ones(len(x0))))
    print(bounds)

    cons = ({'type':'ineq','fun':lambda x: 0})

    output = minimize(min_func,x0,args = (K, S, lambda_, L),method = 'Nelder-Mead',bounds = bounds,constraints = cons)

    P_lambda = output['x']

    return P_lambda

    
def gaussian(r,sigma,mu):
    '''Return Gaussian Distribution from given distance array, standard deviation, and mean distance

    .. math::
        \\frac{1}{\sqrt{2 \pi {\sigma}^2}} e^{-{(r-\mu)}^2/(2\sigma^2)}

    Args:
        r (numpy.ndarray): Numpy array of distance values
        sigma (float): Standard deviation
        mu (float): Mean distance

    Returns:
        gaussian_dist (numpy.ndarray): Gaussian distribution
    '''
    gaussian_dist = (1./(np.sqrt(2*np.pi*(sigma**2.)))) * np.exp(-1*(r-mu)**2./(2.*(sigma**2.)))
    return gaussian_dist

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

