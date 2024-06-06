import numpy as np
from scipy.stats import norm
from pathlib import Path

# Create a random number generator that can be used throughout P-LOOP
# could also seed this generator if they want to fix the random numbers
rng = np.random.default_rng()


def safe_cast_to_array(in_array):
    """
    Attempts to safely cast the input to an array. Takes care of border cases

    Args:
        in_array (array or equivalent): The array (or otherwise) to be converted to a list.

    Returns:
        array : array that has been squeezed and 0-D cases change to 1-D cases

    """

    out_array = np.squeeze(np.array(in_array))

    if out_array.shape == ():
        out_array = np.array([out_array[()]])

    return out_array


def save_archive_dict(archive_dir:Path, save_dict, save_name):
    np.save(archive_dir/save_name, save_dict)

def load_archive_dict(archive_dir:Path, save_name):
    return np.load(archive_dir / save_name).item()
