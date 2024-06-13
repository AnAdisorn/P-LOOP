# %%
import subprocess
import numpy as np
from pathlib import Path
import time

# %%
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
    with open(Path.cwd() / "run.sh", "w") as f:
        f.write(slurm)

    sbatch_stdout = subprocess.run(
        ["sbatch", "run.sh"], capture_output=True, text=True
    ).stdout
    # Extract job id from submit message
    slurm_job_id = sbatch_stdout.split()[-1]
    output_file = results_dir/slurm_job_id/"output.npy"

    # Get output from results folder
    while True:
        if output_file.is_file(): # output file exists now
            output = np.load(output_file)
            return output
        time.sleep(1) # not get the file sleep a little bit

if __name__ == "__main__":
    cwd = Path.cwd()
    slurm_template_path = cwd/"template.sh"
    project_dir = cwd/"project"
    results_dir = cwd/"results"
    output = send_to_cluster(slurm_template_path, "test", project_dir, "python3 code.py", results_dir)
    print(output)
