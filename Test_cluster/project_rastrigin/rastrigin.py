import numpy as np

def rastrigin(params):
    """
    Rastrigin function for 2D optimization.

    Args:
      x: A float representing the x-coordinate.
      y: A float representing the y-coordinate.

    Returns:
      The value of the Rastrigin function at the given (x, y) point.
    """
    x, y = params
    A = 10
    return A * 2 + x**2 - A * np.cos(5 * np.pi * x) + y**2 - A * np.cos(5 * np.pi * y)