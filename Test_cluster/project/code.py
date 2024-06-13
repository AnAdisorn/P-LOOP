import numpy as np
from pathlib import Path

result_dir = Path.cwd() / "result"
#%%
def foo():
    save_path = result_dir / "output.npy"
    np.save(save_path, np.zeros(1))

if __name__ == "__main__":
    foo()
