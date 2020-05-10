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
path = 'path/to/B12MPS/module'
deer.load_elexsys(path)
```

### Performing Tikhonov Regularization ###

```python
import pyDEER as deer

trace = deer.deer_trace(t,r)
```
