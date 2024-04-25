from interfaces import ManagerInterface, WorkerInterface
from learners import Learner, GaussianProcessLearner
from utils import rng
import datetime


class Controller:
    """Base class for all Controller"""

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
        **kwargs,
    ):
        # Halting options.
        self.max_num_runs = float(max_num_runs)
        self.target_cost = float(target_cost)
        self.max_num_runs_without_better_params = float(
            max_num_runs_without_better_params
        )
        self.max_duration = float(max_duration)

        # Variables that are included in the controller
        self.num_in_costs = 0
        self.best_cost = float("inf")
        self.best_uncer = float("nan")
        self.best_index = float("nan")
        self.best_params = float("nan")

        # Variables that are used internally
        self.halt_reasons = []
        self.num_last_best_cost = 0
        self.curr_param = None
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

        # Remaining kwargs
        self.remaining_kwargs = kwargs

        # Start timer
        self.start_datetime = datetime.datetime.now()

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
        # single param
        if params.ndim == 1:
            param_dict = {"params": params}
            self.params_out_queue.put(param_dict)
        # multiple params
        else:
            for param in params:
                param_dict = {"params": param}
                self.params_out_queue.put(param_dict)

    def _get_cost_in_dict(self):
        self.num_in_costs += 1
        param, cost_dict = self.costs_in_queue.get()
        self.curr_param = param
        self.curr_cost = float(cost_dict.pop("cost", float("nan")))
        self.curr_uncer = float(cost_dict.pop("uncer", 0))
        self.curr_bad = bool(cost_dict.pop("bad", False))
        self.curr_run_index = int(cost_dict.pop("run_index", None))
        self.curr_extras = cost_dict

    def _send_to_learner(self):
        """
        Send the latest cost info the the learner.
        """
        if self.curr_bad:
            cost = float("inf")
        else:
            cost = self.curr_cost
        message = (
            self.curr_param,
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
