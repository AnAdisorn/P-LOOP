# Imports for python 2 compatibility
from __future__ import absolute_import, division, print_function

__metaclass__ = type

# Imports for M-LOOP
import mloop.interfaces as mli
import mloop.controllers as mlc
import mloop.visualizations as mlv

# Other imports
import numpy as np
import time


# Declare your custom class that inherits from the Interface class
class CustomInterface(mli.Interface):

    # Initialization of the interface, including this method is optional
    def __init__(self):
        # You must include the super command to call the parent class, Interface, constructor
        super(CustomInterface, self).__init__()

        # Attributes of the interface can be added here
        # If you want to precalculate any variables etc. this is the place to do it
        # In this example we will just define the location of the minimum
        self.minimum_params = np.array([0, 0.1, -0.1])

    # You must include the get_next_cost_dict method in your class
    # this method is called whenever M-LOOP wants to run an experiment
    def get_next_cost_dict(self, params_dict):
        params = params_dict["params"]
        x, y = params
        # Rastrigin function for 2D optimization.
        A = 10
        cost = (
            A * 2 + x**2 - A * np.cos(5 * np.pi * x) + y**2 - A * np.cos(5 * np.pi * y)
        )
        # There is no uncertainty in our result
        uncer = 0
        # The evaluation will always be a success
        bad = False
        # Add a small time delay to mimic a real experiment
        time.sleep(1)

        # The cost, uncertainty and bad boolean must all be returned as a dictionary
        # You can include other variables you want to record as well if you want
        cost_dict = {"cost": cost, "uncer": uncer, "bad": bad}
        return cost_dict


def main():
    # M-LOOP can be run with three commands

    # First create your interface
    interface = CustomInterface()
    # Next create the controller. Provide it with your interface and any options you want to set
    controller = mlc.create_controller(
        interface,
        max_num_runs=100,
        num_params=2,
        min_boundary=[-1, -1],
        max_boundary=[1, 1],
        cost_has_noise=False,
    )
    # To run M-LOOP and find the optimal parameters just use the controller method optimize
    controller.optimize()

    # The results of the optimization will be saved to files and can also be accessed as attributes of the controller.
    print("Best parameters found:")
    print(controller.best_params)

    # You can also run the default sets of visualizations for the controller with one command
    mlv.show_all_default_visualizations(controller)


# Ensures main is run when this code is run as a script
if __name__ == "__main__":
    main()
