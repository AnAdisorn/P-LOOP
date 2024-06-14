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

def load_archive_dict(archive_dir, save_name):
    archive_dir = Path(archive_dir)
    return np.load(archive_dir / save_name, allow_pickle=True).item()

# Cluster
def send_to_cluster(
    params_dict, slurm_template_path, job_name, project_dir, execution_command, results_dir, slurm_dir=None
):
    """
    This function submits a job to a cluster using a provided SLURM template.

    Args:
        params_dict (dictionary): A dictionary containing the parameters and run's index.
        slurm_template_path (str): Path to the SLURM template file.
        job_name (str): Name for the submitted job.
        project_dir (str): Path to the project directory.
        execution_command (str): The command to execute on the cluster.
        results_dir (str): Path to the directory where results will be saved.
        slurm_dir (str, optional): Path to the directory for SLURM errors and outputs. Defaults to None.

    Returns:
        np.ndarray: The loaded output file as a NumPy array.
    """
    # Extract params and run_index from params_dict
    params, run_index = params_dict["params"], params_dict["run_index"]

    # Convert all directories to Path type for easier manipulation
    slurm_template_path = Path(slurm_template_path)
    if slurm_dir is None:
        # If no slurm_dir provided, create a 'slurms' directory in the current working directory
        slurm_dir = Path.cwd() / "slurms"
    else:
        slurm_dir = Path(slurm_dir)  # Convert slurm_dir to Path type as well
    project_dir = Path(project_dir)
    results_dir = Path(results_dir)

    # Create run_dir to contain run.sh and params.npy
    run_dir = Path.cwd() / f"run_{run_index}"
    run_dir.mkdir()
    run_sh = run_dir / "run.sh"

    # Covert params to params.npy, this will be copies to cluster
    params_file = run_dir/"params.npy"
    np.save(params_file, params)

    # Configure job parameters
    # Read the contents of the SLURM template file
    with open(slurm_template_path, "r") as f:
        slurm = f.read()

    # Replace placeholders in the template with actual values
    slurm = slurm.replace("<job_name>", job_name)
    slurm = slurm.replace("<slurm_dir>", str(slurm_dir))
    slurm = slurm.replace("<project_dir>", str(project_dir))
    slurm = slurm.replace("<run_dir>", str(run_dir))
    slurm = slurm.replace("<execution_command>", execution_command)
    slurm = slurm.replace("<results_dir>", str(results_dir))

    # Write the configured script to a 'run.sh' file in the run_{run_index} directory
    with open(run_sh, "w") as f:
        f.write(slurm)

    # Submit the job using sbatch and capture the standard output
    sbatch_stdout = subprocess.run(
        ["sbatch", str(run_sh)], capture_output=True, text=True
    ).stdout

    # Extract the job ID from the last element of the sbatch output after splitting on spaces
    slurm_job_id = sbatch_stdout.split()[-1]

    # Create the output directory path based on the job ID
    output_dir = results_dir / slurm_job_id

    # Define the path to the expected output file
    output_file = output_dir / "output.npy"

    # Loop until the output file is available
    file_not_found_count = 0
    while True:
        if output_dir.is_dir():  # Check if the output directory exists
            if output_file.is_file():  # Check if the file is completely copied
                output = np.load(output_file)  # Load the output as a NumPy array
                # Move the 'run.sh' script and params.npy to the output directory for better organization
                run_sh.rename(output_dir / "run.sh")
                params_file.rename(output_dir / "params.npy")
                # Remove run directory
                run_dir.rmdir()
                return output
            else:
                if file_not_found_count > 3:
                    raise NameError(f"No such file or directory: {str(output_file)}")
                else:
                    file_not_found_count += 1
                    time.sleep(1)  # Wait a bit, maybe the file is copying
        else:
            # If the file isn't there yet, wait for a bit before checking again
            time.sleep(1)
