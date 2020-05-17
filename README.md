# pyDEER #

pyDEER is a python package for Double Electron-Electron Resonance.

The source code for pyDEER is available [here](https://github.com/tkellerBridge12/pyDEER).

The complete documentation for pyDEER is available [here](https://pydeer.readthedocs.io/).

### Installing pyDEER ###

```console
python -m pip install pyDEER
```

### Python Requirements ###

* Python2 (>= 2.7)
* Python3 (>= 3.6)

### Required Modules ###

* scipy
* numpy

```console
python -m pip install scipy numpy
```

### Importing ELEXSYS Data ###

```python
from matplotlib.pylab import *
import pyDEER as deer

# Define path to data
path = './data/20170602_NR119_test/DEER_NR119_55ave'

# Import data
t, data = deer.load_elexsys(path)

# Plot data
figure()
plot(t, data)
xlabel('Time (ns)')
ylabel('Signal (a.u.)')
show()
```

### Performing Tikhonov Regularization ###

```python
import numpy as np
from matplotlib.pylab import *
import pyDEER as deer

# Define time and distance axes
t = np.r_[-100e-9:5e-6:500j]
r = np.r_[1.5e-9:10e-9:100j]

# Generate Kernel Matrix
K = deer.kernel(t, r, angles = 1000)

# Simulate Gaussian P(r)
P_gauss = deer.gaussian(r, 0.2e-9, 4e-9)

# Calculate DEER trace from Gaussian P(r)
S = np.dot(K, P_gauss)

# Add noise to DEER trace
S_noisy = deer.add_noise(S, 0.1)

# Perform Tikhonov Regularization
P_lambda = deer.tikhonov(K, S_noisy, lambda_ = 1.0)

# Calculate Fit of DEER trace
S_fit = np.dot(K, P_lambda)

# Plot Result
figure()
plot(t*1e9, S_noisy, label = 'data')
plot(t*1e9, S_fit, label = 'Tikhonov')
xlabel('Time (ns)')
ylabel('Signal (a.u.)')
legend()

figure('P(r)')
plot(r*1e9, P_gauss, label = 'Exact')
plot(r*1e9, P_lambda, label = 'Tikhonov')
xlabel('r (nm)')
ylabel('P(r)')
legend()
show()
```
