from controllers import GaussianProcessController
from learners import GaussianProcessLearner
from interfaces import ManagerInterface, WorkerInterface
import multiprocessing as mp

learner = GaussianProcessLearner(
    num_params=2,
    min_boundary=[-1, -1],
    max_boundary=[1, 1],
)
workerinterface = WorkerInterface
controller = GaussianProcessController(
    workerinterface=workerinterface,
    num_workers = 4,
    num_params=2,
    min_boundary=[-1, -1],
    max_boundary=[1, 1],
    cost_has_noise=False,
)
controller.learner._update_run_data_attributes([0, 0], 0, 0, 0)
controller.learner.fit_gaussian_process()
print(controller.learner.gaussian_process)
print(controller._first_params())
