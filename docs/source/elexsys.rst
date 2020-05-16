=======
elexsys
=======


Functions
---------

.. automodule:: pyDEER.elexsys
   :members:

Example - elexsys
-----------------

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
