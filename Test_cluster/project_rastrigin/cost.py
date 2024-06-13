from rastrigin import rastrigin
import argparse
import numpy as np
from pathlib import Path

# Get input
params = np.load("params.npy")

# Calculate cost
cost = rastrigin(params)

# Save to result !!! MUST SAVE RESULT THIS WAY !!!
np.save(Path.cwd()/"result/output.npy", cost)



