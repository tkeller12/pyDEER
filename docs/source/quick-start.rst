=================
Quick-Start Guide
=================

Importing pyDEER
----------------

Once installed, pyDEER can be imported as follows:

.. code-block:: python

    import pyDEER as deer

Importing Elexsys Data
----------------------

Bruker Elexsys data can be imported, to import Bruker data:

.. code-block:: python

    import pyDEER as deer

    filename = 'deer_data'

    t, data = import_elexsys(filename)

Generating Simulated Data
-------------------------

For test purposes, it can be useful to simulate the DEER signal due to a Gaussian distance distribution. This can be done as follows:

.. code-block:: python

    import pyDEER as deer

    from matplotlib.pylab import *
    import numpy as np

    t = np.r_[-200e-9:6e-6:200j]

    r = np.r_[1.5e-9:10e-9:100j]

    # Generate Kernel Matrix
    K = deer.kernel(t, r)

    # Generate Gaussian P(r)
    sigma = 0.2e-9
    mu = 4e-9
    P_gauss = deer.gaussian(r, sigma, mu)

    # Calculate DEER Trace
    S = np.dot(K,P_gauss)

    # Add Noise to Data
    S_noisy = deer.add_noise(S, 0.1)

    figure('Simulated DEER Data')
    plot(t*1e9, S, label = 'original')
    plot(t*1e9, S_noisy, label = 'noisy')
    xlabel('Time (ns)')
    ylabel('Signal (a.u.)')
    legend()
    show()

Tikhonov Regularization
-----------------------

Let's take the DEER trace we generated in the last section and perform Tikhonov regularization.

.. code-block:: python

    # Calculate P(r) from Tikhonov regularization
    P_lambda = deer.tikhonov(K, S_noisy, lambda_ = 1.)

    # Calculate Time-domain DEER trace for fit
    S_fit = np.dot(K, P_lambda)

    figure('Tikhonov P(r)')
    plot(r*1e9, P_gauss, label = 'Exact')
    plot(r*1e9, P_lambda, label = 'Tikhonov')
    xlabel('r (nm)')
    ylabel('P(r)')
    legend()

    figure('Tikhonov Fit')
    plot(t*1e9, S_noisy, label = 'data')
    plot(t*1e9, S_fit, label = 'fit')
    xlabel('Time (ns)')
    ylabel('Signal (a.u.)')
    legend()
    show()

Model Free Regularization 
-------------------------

Model free regularization will minimize the same functional as Tikhonov regularization, except with non-negative constraints. In this case, there is no analytical solution to the minimization problem, so a minimization algorithm must be used. 

.. code-block:: python

    # Calculate P(r) from Tikhonov regularization
    P_model_free = deer.model_free(K, S_noisy, lambda_ = 1.)

    # Calculate Time-domain DEER trace for fit
    S_model_free = np.dot(K, P_model_free)

    figure('Model Free P(r)')
    plot(r*1e9, P_gauss, label = 'Exact')
    plot(r*1e9, P_model_free, label = 'Model Free')
    xlabel('r (nm)')
    ylabel('P(r)')
    legend()

    figure('Model Free Fit')
    plot(t*1e9, S_noisy, label = 'data')
    plot(t*1e9, S_model_free, label = 'fit')
    xlabel('Time (ns)')
    ylabel('Signal (a.u.)')
    legend()
    show()

Tikhonov L-curve
----------------

To optimize the regularization parameter, a Tikhonov L-curve can be generated:

.. code-block:: python
    
    # Define array of lambda values
    lambda_array = np.logspace(-2,2,100)

    residual_norm, solution_norm = deer.L_curve(K, S_noisy, lambda_array)

    figure('L-curve')
    plot(residual_norm, solution_norm)
    xlabel('Residual Norm')
    ylabel('Solution Norm')

    show()

Gaussian Based Model
--------------------

A Gaussian based model can also be used for fitting.

.. code-block:: python

    P_fit_gauss, x = deer.model_gaussian(K, S_noisy, r)

    # Gaussian Parameters
    A = x['A']
    sigma = x['sigma']
    mu = x['mu']

    # Gaussian based P(r)
    S_gauss = np.dot(K, P_gauss)

    figure('Gaussian P(r)')
    plot(r*1e9, P_gauss, label = 'Exact')
    plot(r*1e9,P_fit_gauss,label = 'Gaussian')
    xlabel('r (nm)')
    xlabel('P(r)')
    legend()

    figure('Gaussian Fit')
    plot(t*1e9, S_noisy, label = 'data')
    plot(t*1e9,S_gauss,label = 'Gaussian')
    xlabel('Time (ns)')
    xlabel('Signal (a.u.)')
    legend()
    show()

