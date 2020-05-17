==============
elexsys Module
==============

The elexsys module contains functions for importing elexsys data.

Functions
---------

.. automodule:: pyDEER.elexsys
   :members:

Example - 1d elexsys data
-------------------------

.. code-block:: python

    import numpy as np
    from matplotlib.pylab import *
    import pyDEER as deer

    filename = './data/20170602_NR119_test/DEER_NR119_55ave'

    t, data = deer.load_elexsys(filename)

    figure()
    plot(t, data)
    xlabel('Time (ns)')
    ylabel('Signal (a.u.)')
    show()

Example - 2d elexsys data
-------------------------

.. code-block:: python

    import numpy as np
    from matplotlib.pylab import *
    import pyDEER as deer

    filename = './data/20181119_Cu-PAGE/Cu-PAGE_hyscore_11270G_150d1'

    x, hyscore = deer.load_elexsys(filename)

    # For 2d data, first output is a list of axes
    t1 = x[0]
    t2 = x[1]

    # 2d image plot
    figure()
    imshow(np.real(hyscore), aspect = 'auto', origin = 'lower', extent = [t1[0], t1[-1], t2[0], t2[-1]])
    xlabel('t1 (ns)')
    ylabel('t2 (ns)')
    show()
