import unittest
import numpy as np
from math import *

from noiseless_prediction.student_main import f, get_F, unpack_X


class KalmanTest(unittest.TestCase):
    def test_unpack_X(self):
        X = np.matrix([1, 2, 3, 4, 5]).T
        res = unpack_X(X)
        self.assertEqual(res, (1, 2, 3, 4, 5))

    def test_f(self):
        X = np.matrix([1, 2, 3, 3 * pi / 2, pi]).T
        res = f(X, dt=1)
        b = np.allclose(res, np.matrix((-2, 2, 3, pi / 2, pi)).T)
        self.assertTrue(b)

    def test_get_F(self):
        X = np.matrix([1, 2, 3, 3 * pi / 2, pi]).T
        res = get_F(X, dt=1)
        expected = np.matrix([
            [1, 0, -1, 0, 0],
            [0, 1, 0, 3, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 0, 1, 1],
            [0, 0, 0, 0, 1],
        ])
        b = np.allclose(res, np.matrix(expected))
        self.assertTrue(b)


if __name__ == '__main__':
    unittest.main()
