from interfaces import ManagerInterface, WorkerInterface
from learners import Learner, GaussianProcessLearner
from utils import rng, save_archive_dict, load_archive_dict
import logging
import datetime
from pathlib import Path


class Controller:
    """Base class for all Controller"""

    _DEFAULT_ARCHIVE_DIR = Path.cwd() / "controller_archive"
    _DEFAULT_SAVE_PREFIX = "run_"

    def __init__(
        self,
        workerinterface: WorkerInterface,
        learner: Learner,
        num_params,
        min_boundary,
        max_boundary,
        max_num_runs=float("+inf"),
        target_cost=float("-inf"),
        max_num_runs_without_better_params=float("+inf"),
        max_duration=float("+inf"),
        controller_archive_dir=None,
        **kwargs,
    ):
        # Make logger
        self.log = logging.getLogger(__name__)

        # Halting options.
        self.max_num_runs = float(max_num_runs)
        self.target_cost = float(target_cost)
        self.max_num_runs_without_better_params = float(
            max_num_runs_without_better_params
        )
        self.max_duration = float(max_duration)

        # Variables that are included in the controller
        self.num_in_costs = 0  # number of costs received from interaface
        self.best_params = float("nan")
        self.best_cost = float("inf")
        self.best_uncer = float("nan")
        self.best_run_index = float("nan")
        self.best_extras = float("nan")

        # Variables that are used internally
        # curr-prefix means current data from interface
        self.run_index = 0  # used to count number of parameters sent to interface
        self.halt_reasons = []
        self.num_last_best_cost = 0
        self.curr_params = None
        self.curr_cost = None
        self.curr_uncer = None
        self.curr_bad = None
        self.curr_run_index = None
        self.curr_extras = None

        # Connection with interface
        self.managerinterface = ManagerInterface(
            workerinterface=workerinterface, **kwargs
        )
        self.params_out_queue = self.managerinterface.params_out_queue
        self.costs_in_queue = self.managerinterface.costs_in_queue
        self.end_managerinterface = self.managerinterface.end_event

        # Variables for searching parameters
        self.num_params = num_params
        self.min_boundary = min_boundary
        self.max_boundary = max_boundary

        # Connection with learner
        self.learner = learner(
            num_params=num_params,
            min_boundary=min_boundary,
            max_boundary=max_boundary,
            **kwargs,
        )
        self.learner_params_queue = self.learner.params_out_queue
        self.learner_costs_queue = self.learner.costs_in_queue
        self.end_learner = self.learner.end_event

        # Configure archive
        if controller_archive_dir is None:
            self.archive_dir = self._DEFAULT_ARCHIVE_DIR
            if self.archive_dir.is_dir():
                self.log.error(
                    "Controller archive directory already exists, terminate as there is risk in undecided overwrite"
                )
                raise ValueError
        else:
            self.archive_dir = Path(controller_archive_dir)
            if self.archive_dir.is_dir():
                self.log.info(
                    "Given controller archive directory already exists, load all the attributes"
                )
                self.load_archive()

        # Remaining kwargs
        self.remaining_kwargs = kwargs

        # Start timer
        self.start_datetime = datetime.datetime.now()

    def save_to_archive(self):
        """save run attributes to archive"""
        save_dict = {
            "num_in_costs": self.num_in_costs,
            "run_index": self.run_index,
            # current
            "halt_reasons": self.halt_reasons,
            "num_last_best_cost": self.num_last_best_cost,
            "curr_params": self.curr_params,
            "curr_run_index": self.curr_run_index,
            "curr_cost": self.curr_cost,
            "curr_uncer": self.curr_uncer,
            "curr_bad": self.curr_bad,
            "curr_extras": self.curr_extras,
            # best
            "best_params": self.best_params,
            "best_cost": self.best_cost,
            "best_uncer": self.best_uncer,
            "best_run_index": self.best_run_index,
            "best_extras": self.best_extras,
        }
        save_archive_dict(
            self.archive_dir,
            save_dict,
            f"{self._DEFAULT_SAVE_PREFIX}{self.num_in_costs}.npy",
        )

    def load_archive(self):
        # search for the latest run
        save_name = sorted(
            [
                f.name
                for f in self.archive_dir.glob(f"{self._DEFAULT_SAVE_PREFIX}*.npy")
            ],
            key=lambda name: int(name.split("_")[-1][0:-4]),
        )[-1]
        # load save dictionary and set to attributes
        load_dict = load_archive_dict(self.archive_dir, save_name)
        for key, item in load_dict.items():
            setattr(self, key, item)

    def check_end_conditions(self):
        """
        Check whether any of the end contions have been met.

        In particular this method check for any of the following conditions:

        * If the number of iterations has reached `max_num_runs`.
        * If the `target_cost` has been reached.
        * If `max_num_runs_without_better_params` iterations in a row have
          occurred without any improvement.
        * If `max_duration` seconds or more has elapsed since `start_datetime`.

        Returns:
            bool: `True`, if the controller should continue or `False` if one or
                more halting conditions have been met and the controller should
                end.
        """
        # Determine how long it has been since self.start_datetime.
        duration = datetime.datetime.now() - self.start_datetime
        duration = duration.total_seconds()  # Convert to seconds.

        # Check all of the halting conditions. Many if statements are used
        # instead of elif blocks so that we can mark if the optimization halted
        # for more than one reason.
        if self.num_in_costs >= self.max_num_runs:
            self.halt_reasons.append("Maximum number of runs reached.")
        if self.best_cost <= self.target_cost:
            self.halt_reasons.append("Target cost reached.")
        if self.num_last_best_cost >= self.max_num_runs_without_better_params:
            self.halt_reasons.append(
                "Maximum number of runs without better params reached."
            )
        if duration > self.max_duration:
            self.halt_reasons.append("Maximum duration reached.")

        # The optimization should only continue if self.halt_reasons is empty.
        return not bool(self.halt_reasons)

    def _start_up(self):
        """Start the learner and interface threads/processes"""
        self.learner.start()
        self.managerinterface.start()

    def _put_params_out_dict(self, params):
        """Put param/params to interface queue"""
        # single params
        if params.ndim == 1:
            param_dict = {"params": params, "run_index": self.run_index}
            self.params_out_queue.put(param_dict)
            self.run_index += 1
        # multiple set of params
        else:
            for param in params:
                param_dict = {"params": param, "run_index": self.run_index}
                self.params_out_queue.put(param_dict)
                self.run_index += 1

    def _get_cost_in_dict(self):
        # update run count
        self.num_in_costs += 1
        self.num_last_best_cost += 1
        # take in costs information from interface via queue
        param_dict, cost_dict = self.costs_in_queue.get()
        self.curr_params = param_dict
        self.curr_run_index = int(param_dict.pop("run_index", None))
        self.curr_cost = float(cost_dict.pop("cost", float("nan")))
        self.curr_uncer = float(cost_dict.pop("uncer", 0))
        self.curr_bad = bool(cost_dict.pop("bad", False))
        self.curr_extras = cost_dict
        # update the best costs attributes
        if self.curr_cost < self.best_cost:
            self.best_params = self.curr_params
            self.best_cost = self.curr_cost
            self.best_uncer = self.curr_uncer
            self.best_run_index = self.curr_run_index
            self.best_extras = self.curr_extras
            # reset number of run after previous best costs
            self.num_last_best_cost = 0

        # backup to archive
        self.save_to_archive()

    def _send_to_learner(self):
        """
        Send the latest cost info the the learner.
        """
        if self.curr_bad:
            cost = float("inf")
        else:
            cost = self.curr_cost
        message = (
            self.curr_params,
            cost,
            self.curr_uncer,
            self.curr_bad,
            self.curr_run_index,
        )
        self.learner_costs_queue.put(message)

    def _first_params(self):
        pass

    def _next_params(self):
        self._send_to_learner()
        return self.learner_params_queue.get()

    def optimize(self):
        self._start_up()
        self._optimization_routine()
        self._shut_down()

    def _optimization_routine(self):
        next_params = self._first_params()
        self._put_params_out_dict(next_params)
        self._get_cost_in_dict()
        while self.check_end_conditions():
            next_params = self._next_params()
            self._put_params_out_dict(next_params)
            self._get_cost_in_dict()

    def _shut_down(self):
        self.end_learner.set()
        self.end_managerinterface.set()
        self.learner.join()
        self.managerinterface.join()


class GaussianProcessController(Controller):
    """Controller with Gaussian Process leanrer (Bayesian optimisation)"""

    def __init__(self, **kwargs):
        super(GaussianProcessController, self).__init__(
            learner=GaussianProcessLearner,
            **kwargs,
        )

    def _first_params(self):
        params = rng.random((self.managerinterface.num_workers, self.num_params))
        return self.learner.params_scaler.inverse_transform(params)
