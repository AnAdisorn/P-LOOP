# P-LOOP: Parallelizable Bayesian Optimisation

Parallelizable Bayesian optimisation framework.

[**TODO: (Optional) Add badges here, e.g., build status, license, package version. Use services like shields.io.**]
<!-- Example: [![Build Status](https://gitlab.pks.mpg.de/fs1/machine-learning/bayesianoptimisation/badges/main/pipeline.svg)](https://gitlab.pks.mpg.de/fs1/machine-learning/bayesianoptimisation/-/commits/main) -->
<!-- Example: [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) -->

## Description

Bayesian Optimisation is a powerful technique for optimizing expensive black-box functions. This project provides a parallelizable implementation designed to significantly speed up parameter optimization, especially for tasks with long evaluation times.

### Features (Optional)

*   Parallel execution of objective function evaluations via worker processes.
*   Gaussian Process based learner using `scikit-learn`.
*   Expected Improvement acquisition function (adaptable).
*   Asynchronous communication between controller, learner, and workers.
*   Archiving of controller and learner states for resuming runs.
*   Support for initial seed parameters.
*   Flexible `WorkerInterface` for defining custom experiments/simulations.
*   Optional SLURM cluster integration helper (`utils.send_to_cluster`).

## Installation

**Prerequisites:**

*   Python >= 3.8
*   pip (or your preferred package manager like conda)
*   Core dependencies (typically installed automatically via `pip`):
    *   `numpy>=1.20`
    *   `scipy>=1.7`
    *   `scikit-learn>=1.0`

**Steps:**

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/AnAdisorn/P-LOOP.git # Or your specific repo URL
    cd P-LOOP
    ```

2.  **Install the package and its dependencies:**
    *   **Recommended (for development):** Install in editable mode. This links the installation to your source code, so changes are reflected immediately.
        ```bash
        pip install -e .
        ```
    *   **Standard Installation:**
        ```bash
        pip install .
        ```
    *   **(Alternative) Using requirements file (if you create one):**
        ```bash
        # You would need to create a requirements.txt file first
        # pip freeze > requirements.txt
        # Then install using:
        # pip install -r requirements.txt
        ```

## Usage

Here's a basic example of how to use P-LOOP:

```python
import time
import numpy as np
from ploop import GaussianProcessController, WorkerInterface
from ploop.utils import rng # For random number generation consistent with ploop

# 1. Define your custom worker by inheriting from WorkerInterface
class MyExpensiveWorker(WorkerInterface):
    """
    Example worker that simulates an expensive function (e.g., Branin).
    """
    def get_next_cost_dict(self, params_dict):
        params = params_dict['params']
        run_index = params_dict['run_index']
        print(f"Worker {self.pid}: Received job for run {run_index}, params: {params}")

        # Simulate work
        time.sleep(rng.uniform(0.5, 2.0)) # Simulate variable evaluation time

        # Example: Branin function (minimization target)
        x1 = params[0]
        x2 = params[1]
        a = 1.
        b = 5.1 / (4. * np.pi**2)
        c = 5. / np.pi
        r = 6.
        s = 10.
        t = 1. / (8. * np.pi)
        term1 = a * (x2 - b * x1**2 + c * x1 - r)**2
        term2 = s * (1 - t) * np.cos(x1)
        cost = term1 + term2 + s

        print(f"Worker {self.pid}: Finished job for run {run_index}, cost: {cost:.4f}")
        # Return cost (required) and optionally uncertainty or bad flag
        return {'cost': cost}

# 2. Define optimization parameters
num_params = 2
min_boundary = np.array([-5, 0])
max_boundary = np.array([10, 15])
num_workers = 4 # Number of parallel workers
max_runs = 25   # Total evaluations

# 3. Setup and run the controller
# Note: The controller creates archive directories by default.
controller = GaussianProcessController(
    workerinterface=MyExpensiveWorker, # Pass the class, not an instance
    num_params=num_params,
    min_boundary=min_boundary,
    max_boundary=max_boundary,
    num_workers=num_workers,
    max_num_runs=max_runs,
    # Optional: Add initial seeds if desired
    # seeds=np.array([[0, 0], [1, 1]]),
    # Optional: Specify target cost to stop early
    # target_cost=0.5,
)

print("Starting optimization...")
controller.optimize()
print("Optimization finished.")

# 4. Access results
print(f"\n--- Results ---")
print(f"Best parameters found: {controller.best_params}")
print(f"Best cost found: {controller.best_cost:.4f}")
print(f"Found at run index: {controller.best_run_index}")
print(f"Total evaluations: {controller.num_in_costs}")
print(f"Halt reasons: {controller.halt_reasons}")

# You can also inspect the learner's data (requires accessing internal state)
# print("\nLearner Data:")
# print(f"All Params:\n{controller.learner.all_params}")
# print(f"All Costs:\n{controller.learner.all_costs}")

