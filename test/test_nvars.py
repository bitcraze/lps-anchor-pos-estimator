import unittest

import numpy as np

from lpsposest.nvars import nvars


class NvarsTest(unittest.TestCase):

    def test_that_nvars_returns_empty_array_for_empty_input(self):
        # Fixture
        input = np.empty(0)

        # Test
        actual = nvars(input)

        # Assert
        expected = np.empty(0)
        self.assertTrue(np.array_equal(expected, actual))
