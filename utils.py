import numpy as np
from scipy.stats import norm
import subprocess
from pathlib import Path
import time
import os

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
    return np.load(archive_dir / save_name, allow_pickle=True).item()

# Cluster
def send_to_cluster(
    slurm_template_path, job_name, project_dir, execution_command, results_dir, slurm_dir=None
):
    # convert all directories to Path type
    slurm_template_path = Path(slurm_template_path)
    if slurm_dir is None:
        slurm_dir = Path.cwd()/"slurms"
    else:
        slurm_dir = Path(slurm_dir)
    project_dir = Path(project_dir)
    results_dir = Path(results_dir)

    # Configure job parameters
    # read template
    with open(slurm_template_path, "r") as f:
        slurm = f.read()
    # replace template with parameters
    slurm = slurm.replace("<job_name>", job_name)
    slurm = slurm.replace("<slurm_dir>", str(slurm_dir))
    slurm = slurm.replace("<project_dir>", str(project_dir))
    slurm = slurm.replace("<execution_command>", execution_command)
    slurm = slurm.replace("<results_dir>", str(results_dir))
    # write a run.sh file
    run_sh = Path.cwd() / "run.sh"
    with open(run_sh, "w") as f:
        f.write(slurm)

    sbatch_stdout = subprocess.run(
        ["sbatch", "run.sh"], capture_output=True, text=True
    ).stdout
    # Extract job id from submit message
    slurm_job_id = sbatch_stdout.split()[-1]
    output_dir = results_dir/slurm_job_id
    output_file = output_dir/"output.npy"

    # Get output from results folder
    while True:
        if output_file.is_file(): # output file exists now
            output = np.load(output_file)
            os.rename(run_sh, output_dir/"run.sh")
            return output
        time.sleep(1) # not get the file sleep a little bit
