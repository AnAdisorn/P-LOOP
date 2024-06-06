import subprocess

result = subprocess.run(["ls", "-l"], capture_output=True, text=True).stdout

def cost_fn(param_dict):
    params = param_dict["params"]
    run_index = param_dict["run_index"]
    cost_dict = {"cost": (params**2).sum()}
    return param_dict, cost_dict
