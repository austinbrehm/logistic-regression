import unittest
import numpy as np
import main


class TestMain(unittest.TestCase):
    def test_compute_cost_function(self):
        self.assertEqual(main.compute_cost_function(np.array([1]), np.array([1]), 1, 1), 0.12692801104297263)
        self.assertEqual(main.compute_cost_function(np.array([10]), np.array([1]), 1, 1), 0.000016701561318252087)
        self.assertEqual(main.compute_cost_function(np.array([1, 2]), np.array([1, 2]), 1, 1), -1.4122423186916417)

    def test_compute_gradient(self):
        self.assertEqual(main.compute_gradient(np.array([1]), np.array([1]), 1, 1), (-0.11920292202211769,
                                                                                     -0.11920292202211769))
        self.assertEqual(main.compute_gradient(np.array([10]), np.array([1]), 1, 1), (-0.00016701421847953313,
                                                                                      -1.6701421847953313e-05))
        self.assertEqual(main.compute_gradient(np.array([1, 2]), np.array([1, 2]), 1, 1), (-1.10702733418862556,
                                                                                           -0.5833143975998423))

    def test_gradient_descent(self):
        self.assertEqual(main.gradient_descent(np.array([1]), np.array([1]), 1, 1, 0.5, 1), (1.059601461011059,
                                                                                             1.059601461011059,
                                                                                             [0.11344237636122574],
                                                                                             [1.059601461011059],
                                                                                             [1.059601461011059]))


if __name__ == '__main__':
    unittest.main()
