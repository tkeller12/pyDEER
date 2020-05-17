.. pyDEER documentation master file, created by
   sphinx-quickstart on Sat May 09 20:07:32 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

pyDEER Documentation
====================

pyDEER is a python package for processing double electron-electron resonance (DEER) data.

Features:
---------

- Import Elexsys data
- Tikhonov Regularization
- Tikhonov Regularization with non-negative contraints
- Gaussian based model fitting

To install from pip:

.. code-block:: console

    python -m pip install pyDEER


.. toctree::
   :maxdepth: 2
   :caption: Contents:

   install
   quick-start
   tikreg
   elexsys
   awg

References
==========

1.) Yun-Wei Chiang, Peter P. Borbat, Jack H. Freed, The determination of pair distance distributions by pulsed ESR using Tikhonov regularization, J. Magn. Reson. 172 (2005) https://doi.org/10.1016/j.jmr.2004.10.012

2.) Yun-Wei Chiang, Peter P. Borbat, Jack H. Freed, Maximum entropy: A complement to Tikhonov regularization for determination of pair distance distributions by pulsed ESR, J. Magn. Reson. 177 (2005) https://doi.org/10.1016/j.jmr.2005.07.021

3.) Thomas H. Edwards, Stefan Stoll, Optimal Tikhonov regularization for DEER spectroscopy, J. Magn. Reson. 288 (2018) https://doi.org/10.1016/j.jmr.2018.01.021

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
