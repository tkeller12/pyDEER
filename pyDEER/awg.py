import numpy as np

default_number = 43 # default number for saving pulse shape
resolution = 1.e-9 # default pulse shape resolution

# define sech function
np.sech = lambda x: 1./np.cosh(x)

def save_shape(pulse_shape, filename):
    '''Save a numpy array as csv format compatible with Xepr

    Args:
        pulse_shape (numpy.ndarray): Array of pulse shape
        filename (str): Filename to save pulse shape
    '''

    pulse_shape = np.array(pulse_shape)

    with open(filename,'w') as f:
        f.write('begin shape%i "Shape %i"\n'%(int(num),int(num)))

        if pulse_shape.dtype is np.dtype(complex):
            for ix, value in enumerate(pulse_shape):
                f.write('%0.4f,%0.04f\n'%(np.real(pulse_shape[ix]),np.imag(pulse_shape[ix])))
        else:
            for ix, value in enumerate(pulse_shape):
                f.write('%0.4f\n'%(pulse_shape[ix]))

        f.write('end shape%i'%(int(num)))

def load_shape(filename):
    '''Load a pulse shape from csv file

    Args:
        filename (str): Path to file

    Returns: numpy.ndarray
        pulse: Array of pulse shape
    '''


    with open(filename, 'r') as f:
        # Read Preamble
        raw_string = f.read()

    lines = raw_string.strip().split('\n')
    
    pulse_real = []
    pulse_imag = []

    for line in lines:
        if ('begin' in line) or ('end' in line):
            continue
        if ',' in line:
            split_line = line.rsplit(',')
            pulse_real.append(float(split_line[0]))
            pulse_imag.append(float(split_line[1]))
        else:
            pulse_real.append(float(line))
            pulse_imag.append(0.)

    pulse = np.array(pulse_real) + 1j*np.array(pulse_imag)

    return pulse

def adiabatic(tp, BW, beta, resolution = resolution):
    ''' Make Adiabatic Pulse Shape based on Hyperbolic Secant pulse

    .. math::
        \\text{sech} \\left( \\beta (t - \\frac{t_p}{2}) \\right) ^{1+i(\pi BW / \\beta)}

    Args:
        tp (float): pulse length
        BW (float): pulse bandwidth
        beta (float): truncation factor
        resolution (float): pulse resolution

    Returns: tuple

        t (numpy.ndarray): time axes

        pulse (numpy.ndarray): Pulse shape
    '''

    beta = float(beta)/tp
    mu = np.pi*BW/beta
    
    t = np.r_[0.:tp:resolution]

    pulse = (np.sech(beta*(t-0.5*tp)))**(1.+1.j*mu)

    return t, pulse

def chirp(tp, BW, resolution = resolution):
    '''Complex chirp pulse

    .. math::
        e^{i 2 \pi (k/2) (t - t_p/2)^2}

    Args:
        tp (float): Pulse length
        BW (float): Bandwidth of pulse

    Returns: tuple

        t (numpy.ndarray): Time axis

        pulse (numpy.ndarray): Pulse shape
    '''
    k = BW/tp
    t = np.r_[0.:tp:resolution]
    pulse = np.exp(1.j*2.*np.pi*((k/2.)*((t-tp/2.)**2.)))
    return t, pulse

def wurst(tp, N, resolution = resolution):
    '''Real value WURST envelope pulse shape

    .. math::
        1 - \\text{abs} \left( \cos \left( \\frac{\pi}{t_p} (t - \\frac{t_p}{2}) + \\frac{\pi}{2} \\right) \\right) ^N

    Args:
        tp (float): Pulse length
        N (float): exponential 

    Returns: tuple

        t (numpy.ndarray): Time axis of pulse

        pulse (numpy.ndarray): Pulse shape
    '''
    t = np.r_[0.:tp:resolution]
    pulse = (1. - np.abs(np.cos(np.pi*(t-tp/2.)/tp + np.pi/2.))**N) + 0j

    return t,pulse

def gaussian_pulse(tp, sigmas, resolution = resolution):
    '''Gaussian pulse

    .. math::
        e^{- \\frac{1}{2} \left( \\frac{t - t_p/2}{\sigma} \\right)^2}

    Args:
        tp (float): Pulse length
        sigmas (float): Number of standard deviations where pulse is truncated

    Returns: tuple

        t (numpy.ndarray): Time axis

        pulse (numpy.ndarray): Pulse shape
    '''
    sigma = 0.5*tp/sigmas
    t = np.r_[0.:tp:resolution]
    pulse = np.exp(-1.*(t-tp/2.)**2./(2.*(sigma**2.))) + 0j
    return t, pulse

def sq_pulse(tp, t_length = 0., resolution = resolution):
    '''Square pulse

    Args:
        tp (float): Pulse length
        t_length (float): Total length of time axis

    Returns: tuple

        t (numpy.ndarray): Time axis

        pulse (numpy.ndarray): Pulse shape
    '''
    if t_length > tp:
        t = np.r_[0.:t_length:resolution]
        pulse = np.zeros_like(t, dtype = complex)
        for t_ix,t_value in enumerate(t):
            if abs(t_value - t_length/2.) < (tp/2.):
                pulse[t_ix] = 1.
    else:
        t = np.r_[0.:tp:resolution]
        pulse = np.ones_like(t, dtype = complex)
    return t, pulse

def plane_wave(tp, f, resolution = resolution):
    '''Complex plane wave pulse shape

    .. math::
        e^{i 2 \pi f \left( t - \\frac{t_p}{2} \\right) }

    Args:
        tp (float): Pulse length
        f (float): Frequency of plane wave

    Returns: tuple

        t (numpy.ndarray): Time axis

        pulse (numpy.ndarray): Pulse shape
    '''
    t = np.r_[0.:tp:resolution]
    pulse = np.exp(1.j*2.*np.pi*f*(t - tp/2.))
    return t, pulse

def sinc(tp, n, resolution = resolution):
    '''Sinc pulse

    .. math::
        \\frac{\sin \left( \\frac{\pi}{2} (n + 1) x \\right) }{x}

        x = \\frac{t-\\frac{t_p}{2}}{\\frac{t_p}{2}}

    Args:
        tp (float): Pulse length
        n (float): Total sinc lobes, must be odd for full sinc
    n is number of sinc lobes, should be odd for full sinc

    Returns: tuple

        t (numpy.ndarray): Time axis

        pulse (numpy.ndarray): Pulse shape
    '''
    t = np.r_[0.:tp:resolution]
    pulse = np.sin(((n+1.)/2.)*np.pi*(t-tp/2.)/(0.5*tp))/((t-tp/2)/(0.5*tp))

    return t, pulse

if __name__ == '__main__':
    pass
