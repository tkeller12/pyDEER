import unittest
import numpy as np
import sys
sys.path.append('../../')
import pyDEER as deer

class deerTest(unittest.TestCase):
    '''
    '''

    def test_background(self):
        '''
        '''
        t = np.r_[0:10e-6:100j]
        b = deer.background(t,1e-6,0.,1.,d=3.)

        self.assertEqual(1,b[0])

if __name__ == '__main__':
    unittest.main()
