#%%
import subprocess
from pathlib import Path

#%%
def send_to_cluster(slurm_template_path, job_name, project_dir, execution_command, results_dir=None):
    # convert all directories to Path type
    slurm_template_path = Path(slurm_template_path)
    project_dir = Path(project_dir)
    if results_dir is None:
        results_dir = project_dir / "results"
        # check if results directories exists
        if not results_dir.is_dir():
            results_dir.mkdir()
    else:
        results_dir = Path(results_dir)
    
    # Configure job parameters
    # read template
    with open(slurm_template_path, 'r') as f:
        slurm = f.read()
    # replace template with parameters
    slurm = slurm.replace("<job_name>", job_name)
    slurm = slurm.replace("<project_dir>", project_dir)
    slurm = slurm.replace("<execution_command>", execution_command)
    # write a run file
    with open(Path.cwd()/"run.sh", 'w') as f:
        f.write(slurm)
    
    sbatch_stdout = subprocess.run(["sbatch", "run.sh"], capture_output=True, text=True).stdout
    

    


# def cost_fn(param_dict):
#     params = param_dict["params"]
#     run_index = param_dict["run_index"]
#     cost_dict = {"cost": (params**2).sum()}
#     return param_dict, cost_dict
