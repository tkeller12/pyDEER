import numpy as np
import os

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

def background(t,c, d = 3.):
    '''DEER Background function

    .. math::
        e^{-c t^{d/3}}

    Args:
        t (numpy.ndarray): Time axes for background function
        c (float): Constant 
        d (float): dimensionality of background function

    Returns:
        background_signal (numpy.ndarray)
    '''
    background_signal = np.exp(-c*(t**(dimensionality/3.)))
    return background_signal

def tikhonov(K,S,lambda_ = 1.0,L = None):
    '''Perform Tikhonov Regularization

    .. math::
        P_\lambda = {(K^TK + \lambda L^TL)}^{-1} K^TS

    Args:
        K (numpy.ndarray): Kernel Matrix
        S (numpy.ndarray): Experimental DEER trace
        lambda_ (float): Regularization parameter
        operator (None, numpy.ndarray): Tikhonov regularization operator, uses identity if argument is None
    '''
    # Select Real Part
    S = np.real(S)

    # Set Operator for Tikhonov Regularization
    if L == None:
        L = np.eye(np.shape(K)[1])

    P_lambda = np.dot(np.linalg.inv(np.dot(K.T,K)+(lambda_**2.)*np.dot(L.T,L)),np.dot(K.T,S))

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

#def gaussian_dist(r,sigma,mu,pts=100):
#    '''Generate Gaussian distribution
#    Args:
#
#    Returns:
#
#    '''
#    v = np.zeros(len(t))
#
#    for x in r:
#        weight = gaussian(x,sigma,mu)
#        v += weight*signal(x,t)
#    return v

if __name__ == '__main__':
    from matplotlib.pylab import *

    t = np.r_[-0.1e-6:10e-6:200j]
    r = np.r_[1.5e-9:8e-9:500j]

    trace = deer_trace(t,4e-9,angles = 100)
    gaussian_dist = gaussian(r,0.5e-9,5e-9)
    kernel_matrix = kernel(t,r,angles = 2000)

    deer = np.dot(kernel_matrix,gaussian_dist)

    P_calc = tikhonov(kernel_matrix,deer)


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
    plot(t*1e6,deer)
    figure()
    plot(r,P_calc)
    show()
