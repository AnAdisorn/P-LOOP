# %%
# PLOOP
from controllers import GaussianProcessController
from interfaces import WorkerInterface
from utils import send_to_cluster

# Other imports
import numpy as np
from pathlib import Path
import time


# %%
# Declare your custom class that inherits from the Interface class
class CustomInterface(WorkerInterface):

    # Initialization of the interface, including this method is optional
    def __init__(self):
        # You must include the super command to call the parent class, Interface, constructor
        super(CustomInterface, self).__init__()

    # You must include the get_next_cost_dict method in your class
    # this method is called whenever P-LOOP wants to run an experiment
    def get_next_cost_dict(self, params_dict):
        # prepare parameters to send to cluster
        slurm_template_path = Path.cwd() / "Test_cluster/template.sh"
        job_name = "rastrigin"
        project_dir = Path.cwd() / "Test_cluster/project_rastrigin"
        execution_command = f"python3 cost.py"
        results_dir = Path.cwd() / "Test_cluster/results_rastrigin"
        slurms_dir = Path.cwd() / "Test_cluster/slurms_rastrigin"
        # calculate cost in cluster
        cost = send_to_cluster(
            params_dict,
            slurm_template_path,
            job_name,
            project_dir,
            execution_command,
            results_dir,
            slurms_dir,
        )

        # return simple cost_dict
        cost_dict = {"cost": cost}
        return cost_dict


def main():
    # M-LOOP can be run with three commands

    # First create your interface
    interface = CustomInterface
    # Next create the controller. Provide it with your interface and any options you want to set
    controller = GaussianProcessController(
        workerinterface=interface,
        max_num_runs=200,
        num_params=2,
        min_boundary=[-1, -1],
        max_boundary=[1, 1],
        cost_has_noise=False,
        num_workers=10,
    )
    # To run M-LOOP and find the optimal parameters just use the controller method optimize
    controller.optimize()

    # The results of the optimization will be saved to files and can also be accessed as attributes of the controller.
    print("Best parameters found:")
    print(controller.best_params)


# Ensures main is run when this code is run as a script
if __name__ == "__main__":
    main()
