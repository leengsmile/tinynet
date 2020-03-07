import unittest

import numpy as np
from scipy.stats import norm


class TestScipyNormalDist(unittest.TestCase):

    def test_scipy_normal_distribution(self):

        def normal(x):
            return np.exp(- x ** 2 / 2) / np.sqrt(2 * np.pi)

        x = np.linspace(-3, 3, num=5)
        p_scipy = norm.pdf(x)
        p_hand = normal(x)
        print("p scipy: ", p_scipy)
        print("p hand:", p_hand)
        np.testing.assert_allclose(p_scipy, p_hand,)
