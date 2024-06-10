import multiprocessing as mp
import threading
import logging


class ManagerInterface(threading.Thread):

    def __init__(self, workerinterface=None, num_workers=1, **kwargs):

        super(ManagerInterface, self).__init__()

        # Make logger
        self.log = logging.getLogger(__name__)

        # Queues to share with worker
        self.params_out_queue = mp.Queue()
        self.costs_in_queue = mp.Queue()
        self.end_event = mp.Event()

        # Worker Interface
        if workerinterface is None:
            raise ValueError("Worker interface is not provided")
        else:
            self.workerinterface = workerinterface

        self.num_workers = num_workers
        self.workers = []  # list of workers interface

        self.remaining_kwargs = kwargs

    def _set_up(self):
        """Setup and share Queues to Workers"""
        for _ in range(self.num_workers):
            worker = self.workerinterface()
            worker.set_up_queue(self.params_out_queue, self.costs_in_queue, self.end_event)
            self.workers.append(worker)

    def _start_up(self):
        for worker in self.workers:
            worker.start()

    def run(self):
        """The run sequence for the managerinterface."""
        self._set_up()
        self._start_up()


class WorkerInterface(mp.Process):
    """
    A abstract class for worker interfaces which populate the costs_in_queue and read from the params_out_queue.

    Arguments:
        params_out_queue (queue): Queue for parameters to next be run by experiment.
        costs_in_queue (queue): Queue for costs (and other details) that have been returned by experiment.
        end_event (event): Event which triggers the end of the interface.

    """

    def __init__(self):
        super(WorkerInterface, self).__init__()

    def set_up_queue(self, params_out_queue, costs_in_queue, end_event):
        self.params_out_queue = params_out_queue
        self.costs_in_queue = costs_in_queue
        self.end_event = end_event

    def run(self):
        """
        The run sequence for the workerinterface.

        This method does NOT need to be overloaded create a working interface.
        """
        while not self.end_event.is_set():
            # Wait for the next set of parameter values to test.
            params_dict = self.params_out_queue.get()
            param_dict, cost_dict = self.get_next_cost_dict(params_dict)
            # Send the results back to the controller.
            self.costs_in_queue.put((param_dict, cost_dict))

    def get_next_cost_dict(self, param_dict):
        """
        Abstract method.

        This is the only method that needs to be implemented to make a working
        interface.

        Given the parameters the interface must then produce a new cost. This
        may occur by running an experiment or program. If an error is raised by
        this method, the optimization will halt.

        Args:
            params_dict (dictionary): A dictionary containing the parameters
                AND run's index.
                Use `params_dict['params']` and `params_dict['run_index']`
                to access them.

        Returns:
            param_dict (dictionary): return the in-param_dict as reference
            cost_dict (dictionary): The cost and other properties derived from
                the experiment when it was run with the parameters. If just a
                cost was produced provide `{'cost': [float]}`, if you also have
                an uncertainty provide `{'cost': [float], 'uncer': [float]}`.
                If the run was bad you can simply provide `{'bad': True}`.
                For completeness you can always provide all four using
                `{'cost': [float], 'uncer':[float], 'bad': [bool]}`.
                Any extra keys provided will also be saved by the controller.
        """
        pass
