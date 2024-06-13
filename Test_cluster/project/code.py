import numpy as np
from pathlib import Path

result_dir = Path.cwd() / "result"
result_dir.mkdir()


def foo():
    np.save(result_dir / "test", np.zeros(1))
