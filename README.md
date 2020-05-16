# README #

pyDEER is a python package for Double Electron-Electron Resonance

### Requirements ###

* Python3 (>= 3.6)
* numpy

```console
python -m pip install numpy
```

### Importing ELEXSYS Data ###

```python
import pyDEER as deer
path = 'path/to/deer/data/including/filename'
deer.load_elexsys(path)
```

### Performing Tikhonov Regularization ###

```python
import pyDEER as deer
import numpy as np
r = np.r_[1.5e-9:10e-9:100j]
t = np.r_[-100e-9:5e-6,500j]

K = deer.kernel(t,r,angles = 1000)

P_gauss = gaussian(r, 0.2e-9, 4-9)
trace = deer.deer_trace(t,r)

P_lambda = deer.tikhonov(K
```
