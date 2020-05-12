import numpy as np

default_number = 43 # default number for saving pulse shape
resolution = 1.e-9 # default pulse shape resolution

# define sech function
np.sech = lambda x: 1./np.cosh(x)

def save_shape(pulse_shape,filename):
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

    Return:
        pulse (numpy.ndarray): Array of pulse shape
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

def save_shapes(pulse_shapes, filename):
    '''Save a numpy array as csv format compatible with Xepr

    Args:
        pulse_shapes (numpy.ndarray,list): numpy array or list of numpy arrays defining pulse shape
        filename (str): Filename to save data
    '''

    with open(filename,'w') as f:
        num_ix = 0
        for pulse_shape in pulse_shapes:
            pulse_shape = np.array(pulse_shape)
            num=nums[num_ix]
            num_ix+=1

            f.write('begin shape%i "Shape %i"\n'%(int(num),int(num)))

            if pulse_shape.dtype is np.dtype(complex):
                for ix, value in enumerate(pulse_shape):
                    f.write('%0.4f,%0.04f\n'%(np.real(pulse_shape[ix]),np.imag(pulse_shape[ix])))
            else:
                for ix, value in enumerate(pulse_shape):
                    f.write('%0.4f\n'%(pulse_shape[ix]))

            if num_ix == len(pulse_shapes):
                f.write('end shape%i'%(int(num)))
            else:
                f.write('end shape%i\n'%(int(num)))

def adiabatic(tp,BW,beta,resolution = resolution):
    ''' Make Adiabatic Pulse Shape based on Hyperbolic Secant pulse

    Args:
        tp (float): pulse length
        BW (float): pulse bandwidth
        beta (float): 
        resolution (float): pulse resolution

    Returns:
        t (numpy.ndarray): time axes of pulse
    '''

    beta = float(beta)/tp
    mu = np.pi*BW/beta
    
    t = np.r_[0:tp:resolution]

    pulse = (np.sech(beta*(t-0.5*tp)))**(1.+1.j*mu)

    return t, pulse

