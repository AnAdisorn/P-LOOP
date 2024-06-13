# %%
import subprocess
from pathlib import Path


# %%
def send_to_cluster(
    slurm_template_path, job_name, project_dir, execution_command, results_dir
):
    # convert all directories to Path type
    slurm_template_path = Path(slurm_template_path)
    project_dir = Path(project_dir)
    results_dir = Path(results_dir)

    # Configure job parameters
    # read template
    with open(slurm_template_path, "r") as f:
        slurm = f.read()
    # replace template with parameters
    slurm = slurm.replace("<job_name>", job_name)
    slurm = slurm.replace("<project_dir>", project_dir)
    slurm = slurm.replace("<execution_command>", execution_command)
    slurm = slurm.replace("<results_dir>", execution_command)
    # write a run.sh file
    with open(Path.cwd() / "run.sh", "w") as f:
        f.write(slurm)

    sbatch_stdout = subprocess.run(
        ["sbatch", "run.sh"], capture_output=True, text=True
    ).stdout
    print(sbatch_stdout)


if __name__ == "__main__":
    slurm_template_path = (
        "/Users/adisorn/Desktop/Codes/bayesianoptimisation/Test_cluster/template.sh"
    )
    project_dir = (
        "/Users/adisorn/Desktop/Codes/bayesianoptimisation/Test_cluster/project"
    )
    results_dir = (
        "/Users/adisorn/Desktop/Codes/bayesianoptimisation/Test_cluster/results"
    )
    send_to_cluster(slurm_template_path, "test", project_dir, "python3 code.py", results_dir)
