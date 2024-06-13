# %%
from utils import send_to_cluster
import subprocess
import numpy as np
from pathlib import Path
import os
import time

#%%
cwd = Path.cwd()
slurm_template_path = cwd/"Test_cluster/template.sh"
project_dir = cwd/"Test_cluster/project"
results_dir = cwd/"Test_cluster/results"
output = send_to_cluster(slurm_template_path, "test", project_dir, "python3 code.py", results_dir)
print(output)
