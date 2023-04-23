import unittest
import numpy as np
import main


class TestMain(unittest.TestCase):
    def test_compute_cost_function(self):
        self.assertEqual(main.compute_cost_function(np.array([1]), np.array([1]), 1, 1), 0.12692801104297263)
        self.assertEqual(main.compute_cost_function(np.array([10]), np.array([1]), 1, 1), 0.000016701561318252087)
        self.assertEqual(main.compute_cost_function(np.array([1, 2]), np.array([1, 2]), 1, 1), -1.4122423186916417)

    def test_compute_gradient(self):
        pass

    def test_gradient_descent(self):
        pass


if __name__ == '__main__':
    unittest.main()
