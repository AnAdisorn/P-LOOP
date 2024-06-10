# %%
import numpy as np
import multiprocessing as mp
import queue
import scipy.optimize as so
import sklearn.gaussian_process as skg
import sklearn.gaussian_process.kernels as skk
import sklearn.preprocessing as skp
from utils import rng, norm, safe_cast_to_array, save_archive_dict, load_archive_dict
import logging
import datetime
from pathlib import Path


# %%
class Learner(mp.Process):
    "Base class for learner"

    _DEFAULT_SAVE_NAME = "search_storage"

    def __init__(
        self,
        num_params=None,
        min_boundary=None,
        max_boundary=None,
        learner_archive_dir=None,
        **kwargs,
    ):

        super(Learner, self).__init__()

        # Make logger
        self.log = logging.getLogger(__name__)

        # Set up Queues to communicate with controller
        self.params_out_queue = mp.Queue()
        self.costs_in_queue = mp.Queue()
        self.end_event = mp.Event()

        # Configure parameters boundaries
        self.num_params = int(num_params)
        if min_boundary is None:
            min_boundary = np.full(self.num_params, -1.0)
        else:
            min_boundary = np.asarray(min_boundary, dtype=float)
        if max_boundary is None:
            max_boundary = np.full(self.num_params, +1.0)
        else:
            max_boundary = np.asarray(max_boundary, dtype=float)
        self.search_boundary = np.transpose([min_boundary, max_boundary])

        # Storage for seached parameters
        self.all_params = np.array([], dtype=float)
        self.all_costs = np.array([], dtype=float)
        self.all_run_indices = np.array([], dtype=int)
        self.best_params = float("nan")
        self.best_cost = float("inf")
        self.best_index = None
        self.worst_params = float("nan")
        self.worst_cost = -float("inf")
        self.worst_index = None

        # Configure archive
        if learner_archive_dir is None:
            self.archive_dir = self._DEFAULT_ARCHIVE_DIR
            if self.archive_dir.is_dir():
                self.log.error(
                    f"Learner archive directory already exists at {self.archive_dir}, terminate as there is risk in undecided overwrite"
                )
                raise ValueError
            else:
                self.log.info(f"Create lerner archive: {self.archive_dir}")
                self.archive_dir.mkdir()
        else:
            self.archive_dir = Path(learner_archive_dir)
            if self.archive_dir.is_dir():
                self.log.info(
                    "Given learner archive directory already exists, load all the attributes"
                )
                self.load_archive()

        #  Etc
        self.start_datetime = datetime.datetime.now()
        self.remaining_kwargs = kwargs

    def save_archive(self):
        save_dict = {
            "all_params": self.all_params,
            "all_costs": self.all_costs,
            "all_run_indices": self.all_run_indices,
            "best_params": self.best_params,
            "best_cost": self.best_cost,
            "best_index": self.best_index,
            "worst_params": self.worst_params,
            "worst_cost": self.worst_cost,
            "worst_index": self.worst_index,
        }
        save_archive_dict(self.archive_dir, save_dict, f"{self._DEFAULT_SAVE_NAME}.npy")

    def load_archive(self):
        # load save dictionary and set to attributes
        load_dict = load_archive_dict(
            self.archive_dir, f"{self._DEFAULT_SAVE_NAME}.npy"
        )
        for key, item in load_dict.items():
            setattr(self, key, item)

    def _update_run_data_attributes(self, param, cost, uncer, run_index):
        """
        Update attributes that store the results returned by the controller.

        Args:
            params (array): Array of control parameter values.
            cost (float): The cost measured for `params`.
        """
        if self.all_params.size == 0:
            self.all_params = np.asarray([param], dtype=float)
            self.all_costs = np.asarray([cost], dtype=float)
            self.all_run_indices = np.asarray([run_index], dtype=int)
            self.all_uncers = np.asarray([uncer], dtype=float)
        else:
            # params
            params_array = np.asarray([param], dtype=float)
            self.all_params = np.append(self.all_params, params_array, axis=0)
            # cost
            cost_array = np.asarray([cost], dtype=float)
            self.all_costs = np.append(self.all_costs, cost_array, axis=0)
            # run index
            run_index_array = np.asarray([run_index], dtype=int)
            self.all_run_indices = np.append(self.all_run_indices, run_index_array)
            # uncer
            uncer_array = np.asarray([uncer], dtype=float)
            self.all_uncers = np.append(self.all_uncers, uncer_array, axis=0)

        if cost < self.best_cost:
            self.best_cost = cost
            self.best_params = param
            self.best_index = run_index
        if cost > self.worst_cost:
            self.worst_cost = cost
            self.worst_params = param
            self.worst_index = run_index

        # backup to archive
        self.save_archive()


# %%
class GaussianProcessLearner(Learner):

    _DEFAULT_ARCHIVE_DIR = Path.cwd() / "gaussian_learner_archive"
    _DEFAULT_SCALED_LENGTH_SCALE = 1e-1
    _DEFAULT_SCALED_LENGTH_SCALE_BOUNDS = np.array([1e-3, 1e1])
    _DEFAULT_ALPHA = 1e-8

    def __init__(
        self,
        length_scale=None,
        length_scale_bounds=None,
        cost_has_noise=True,
        noise_level=None,
        noise_level_bounds=None,
        update_hyperparameters=True,
        **kwargs,
    ):

        super(GaussianProcessLearner, self).__init__(**kwargs)

        # Internal variable for bias function
        self.search_precision = 1.0e-6
        self.parameter_searches = max(10, self.num_params)
        self.hyperparameter_searches = max(10, self.num_params)
        self.bias_func_cycle = 4
        self.bias_func_uncer_factor = [0.0, 1.0, 2.0, 3.0]
        self.generation_num = self.bias_func_cycle
        self.params_count = 0

        # Scalers for the costs and parameter values
        self.cost_scaler = skp.StandardScaler()
        self.params_scaler = skp.MinMaxScaler().fit(self.search_boundary.T)

        # Storage for parameters
        self.search_params = []
        self.scaled_params = np.array([], dtype=float)
        self.scaled_costs = np.array([], dtype=float)

        # Configure GaussianProcess
        self.gaussian_process = None
        self.cost_has_noise = cost_has_noise
        self.update_hyperparameters = update_hyperparameters
        if length_scale is None:
            self.scaled_length_scale = self._DEFAULT_SCALED_LENGTH_SCALE
            self.length_scale = self._transform_length_scale(
                self.scaled_length_scale, inverse=True
            )
        else:
            self.length_scale = np.asarray(length_scale, dtype=float)
            self.scaled_length_scale = self._transform_length_scale(self.length_scale)
        if length_scale_bounds is None:
            self.scaled_length_scale_bounds = self._DEFAULT_SCALED_LENGTH_SCALE_BOUNDS
            self.length_scale_bounds = self._transform_length_scale_bounds(
                self.scaled_length_scale_bounds, inverse=True
            )
        else:
            self.length_scale_bounds = np.asarray(length_scale_bounds, dtype=float)
            self.scaled_length_scale_bounds = self._transform_length_scale_bounds(
                self.length_scale_bounds
            )
        if cost_has_noise:
            if noise_level is None:
                # Temporarily change to NaN to mark that the default value
                # should be calcualted once training data is available. Using
                # NaN instead of None is necessary in case the archive is saved
                # in .mat format since it can handle NaN but not None.
                self.noise_level = float("nan")
                self.scaled_noise_level = 1.0
            else:
                self.noise_level = float(noise_level)
            if noise_level_bounds is None:
                self.noise_level_bounds = float("nan")
                self.scaled_noise_level_bounds = np.array([1e-8, 1e8])
            else:
                self.noise_level_bounds = safe_cast_to_array(noise_level_bounds)

    def run(self):
        self.log.info("Lerner: start running GP learner")
        while not self.end_event.is_set():
            try:
                self.get_params_and_costs()
                self.fit_gaussian_process()
                next_params = self.find_next_parameters()
                self.params_out_queue.put(next_params)
            except queue.Empty:
                continue

    def get_params_and_costs(self):
        """
        Get the parameters and costs from the queue.
        """
        # First get, block until get something
        (param, cost, uncer, _, run_index) = self.costs_in_queue.get(timeout=1)
        self._update_run_data_attributes(param, cost, uncer, run_index)
        # # If more costs in queue, get them
        # while not self.costs_in_queue.empty():
        #     (param, cost, uncer, run_index) = self.costs_in_queue.get()
        #     self._update_run_data_attributes(param, cost, uncer, run_index)

    def _transform_length_scale(self, length_scale, inverse=False):
        scale_factor = self.params_scaler.scale_
        return scale_factor / length_scale if inverse else scale_factor * length_scale

    def _transform_length_scale_bounds(self, length_scale_bounds, inverse=False):
        if length_scale_bounds.shape == (2,):
            # Single pair
            min_, max_ = length_scale_bounds
            lower_bounds = np.full(self.num_params, min_, dtype=float)
            upper_bounds = np.full(self.num_params, max_, dtype=float)
        else:
            # Multiple pair
            lower_bounds = length_scale_bounds[:, 0]
            upper_bounds = length_scale_bounds[:, 1]

        scaled_lower_bounds = self._transform_length_scale(
            lower_bounds, inverse=inverse
        )
        scaled_upper_bounds = self._transform_length_scale(
            upper_bounds, inverse=inverse
        )
        return np.transpose([scaled_lower_bounds, scaled_upper_bounds])

    def create_gaussian_process(self):
        """
        Create a Gaussian process regressor.

        This function defines and initializes a Gaussian Process (GP) regressor
        object. The GP regressor will be used to model the relationship between
        input features and target values.

        """

        # Define the kernel function
        gp_kernel = skk.RBF(  # Radial Basis Function kernel
            length_scale=self.scaled_length_scale,
            length_scale_bounds=self.scaled_length_scale_bounds,
        )

        # Add white noise kernel if cost has noise
        if self.cost_has_noise:
            white_kernel = skk.WhiteKernel(
                noise_level=self.scaled_noise_level,
                noise_level_bounds=self.scaled_noise_level_bounds,
            )
            gp_kernel += white_kernel  # Combine RBF and white noise kernels
            alpha = self.scaled_uncers**2  # Set alpha based on uncertainties
        else:
            alpha = self._DEFAULT_ALPHA  # Use default alpha if no noise

        # Choose optimizer based on hyperparameter update setting
        if self.update_hyperparameters:
            self.gaussian_process = skg.GaussianProcessRegressor(
                alpha=alpha,
                kernel=gp_kernel,
                n_restarts_optimizer=self.hyperparameter_searches,
            )
        else:
            self.gaussian_process = skg.GaussianProcessRegressor(
                alpha=alpha, kernel=gp_kernel, optimizer=None
            )

    def fit_gaussian_process(self):
        # Scaling params and costs
        self.scaled_params = self.params_scaler.transform(self.all_params)
        self.scaled_costs = self.cost_scaler.fit_transform(
            self.all_costs[:, np.newaxis]
        )[:, 0]
        cost_scaling_factor = float(self.cost_scaler.scale_)
        self.scaled_uncers = self.all_uncers / cost_scaling_factor

        if self.cost_has_noise:
            if np.isnan(self.noise_level):
                # Set noise_level to its default value, which is the variance of
                # the training data, which is equal to the square of the cost
                # scaling factor. This will only happen on first iteration since
                # self.noise_level is overwritten.
                self.noise_level = cost_scaling_factor**2
            if np.any(np.isnan(self.noise_level_bounds)):
                self.noise_level_bounds = np.array([1e-8, 1e8]) * cost_scaling_factor**2

        # Fit Gaussian process
        self.create_gaussian_process()
        self.gaussian_process.fit(self.scaled_params, self.scaled_costs)

    def predict_cost(
        self,
        params,
        perform_scaling=True,
        return_uncertainty=False,
    ):
        """
        Predict the cost for `params` using `self.gaussian_process`.

        This method also optionally returns the uncertainty of the predicted
        cost.

        By default (with `perform_scaling=True`) this method will use
        `self.params_scaler` to scale the input values and then use
        `self.cost_scaler` to scale the cost back to real/unscaled units. If
        `perform_scaling` is `False`, then this scaling will NOT be done. In
        that case, `params` should consist of already-scaled parameter values
        and the returned cost (and optional uncertainty) will be in scaled
        units.

        Args:
                params (array): A 1D array containing the values for each parameter.
                        These should be in real/unscaled units if `perform_scaling` is
                        `True` or they should be in scaled units if `perform_scaling` is
                        `False`.
                perform_scaling (bool, optional): Whether or not the parameters and
                        costs should be scaled. If `True` then this method takes in
                        parameter values in real/unscaled units then returns a predicted
                        cost (and optionally the predicted cost uncertainty) in
                        real/unscaled units. If `False`, then this method takes
                        parameter values in scaled units and returns a cost (and
                        optionally the predicted cost uncertainty) in scaled units. Note
                        that this method cannot determine on its own if the values in
                        `params` are in real/unscaled units or scaled units; it is up to
                        the caller to pass the correct values. Defaults to `True`.
                return_uncertainty (bool, optional): This optional argument controls
                        whether or not the predicted cost uncertainty is returned with
                        the predicted cost. The predicted cost uncertainty will be in
                        real/unscaled units if `perform_scaling` is `True` and will be
                        in scaled units if `perform_scaling` is `False`. Defaults to
                        `False`.

        Returns:
                cost (float): Predicted cost at `params`. The cost will be in
                        real/unscaled units if `perform_scaling` is `True` and will be
                        in scaled units if `perform_scaling` is `False`.
                uncertainty (float, optional): The uncertainty of the predicted
                        cost. This will be in the same units (either real/unscaled or
                        scaled) as the returned `cost`. The `cost_uncertainty` will only
                        be returned if `return_uncertainty` is `True`.
        """
        # Reshape to 2D array as the methods below expect this format.
        params_ndim = params.ndim
        if params_ndim == 1:
            params = params[np.newaxis, :]

        # Scale the input parameters if set to do so.
        if perform_scaling:
            scaled_params = self.params_scaler.transform(params)
        else:
            scaled_params = params

        # Generate the prediction using self.gaussian_process.
        predicted_results = self.gaussian_process.predict(
            scaled_params,
            return_std=return_uncertainty,
        )
        if return_uncertainty:
            scaled_cost, scaled_uncertainty = predicted_results
        else:
            scaled_cost = predicted_results

        # Un-scale the cost if set to do so.
        if perform_scaling:
            cost = self.cost_scaler.inverse_transform(
                scaled_cost.reshape(-1, 1),
            )
        else:
            cost = scaled_cost
        cost = cost.reshape(-1)  # return to 1D

        # Un-scale the uncertainty if set to do so.
        if return_uncertainty:
            if perform_scaling:
                cost_scaling_factor = self.cost_scaler.scale_
                uncertainty = scaled_uncertainty * cost_scaling_factor
            else:
                uncertainty = scaled_uncertainty
            uncertainty = uncertainty.reshape(-1)

        # Change output in case of single params input (minimisation process)
        if params_ndim == 1:
            cost = cost[0]
            if return_uncertainty:
                uncertainty = uncertainty[0]

        # Return the requested results.
        if return_uncertainty:
            return cost, uncertainty
        else:
            return cost

    def update_bias_function(self):
        """
        Set the constants for the cost bias function.
        """
        self.uncer_bias = self.bias_func_uncer_factor[
            self.params_count % self.bias_func_cycle
        ]

    def predict_biased_cost(self, params, perform_scaling=True):
        """
        Predict the biased cost at the given parameters.
        AKA Acquisition function

        The biased cost is a weighted sum of the predicted cost and the
        uncertainty of the prediced cost. In particular, the bias function is:
            `biased_cost = cost_bias * pred_cost - uncer_bias * pred_uncer`

        Args:
            params (array): A 1D array containing the values for each parameter.
                These should be in real/unscaled units if `perform_scaling` is
                `True` or they should be in scaled units if `perform_scaling` is
                `False`.
            perform_scaling (bool, optional): Whether or not the parameters and
                biased costs should be scaled. If `True` then this method takes
                in parameter values in real/unscaled units then returns a biased
                predicted cost in real/unscaled units. If `False`, then this
                method takes parameter values in scaled units and returns a
                biased predicted cost in scaled units. Note that this method
                cannot determine on its own if the values in `params` are in
                real/unscaled units or scaled units; it is up to the caller to
                pass the correct values. Defaults to `True`.

        Returns:
            pred_bias_cost (float): Biased cost predicted for the given
                parameters. This will be in real/unscaled units if
                `perform_scaling` is `True` or it will be in scaled units if
                `perform_scaling` is `False`.
        """
        # Determine the predicted cost and uncertainty.
        cost, uncertainty = self.predict_cost(
            params,
            perform_scaling=perform_scaling,
            return_uncertainty=True,
        )

        # Calculate the biased cost.

        ## Upper confident bound (UCB)
        # biased_cost = cost - self.uncer_bias * uncertainty

        ## Probability of improvement
        # if perform_scaling:
        #     best_cost = self.best_cost
        # else:
        #     best_cost = self.cost_scaler.transform([[self.best_cost]])[0, 0]
        # biased_cost = -norm.cdf((best_cost - cost) / uncertainty)

        ## Expected improvement as biased cost
        if perform_scaling:
            best_cost = self.best_cost
        else:
            best_cost = self.cost_scaler.transform([[self.best_cost]])[0, 0]
        delta = best_cost * (1 + 0.0 / 100.0) - cost
        ratio = delta / uncertainty
        biased_cost = -(delta * norm.cdf(ratio) + uncertainty * norm.pdf(ratio))

        return biased_cost

    def update_search_params(self):
        """
        Update the list of parameters to use for the next search.
        """
        self.search_params = []
        self.search_params.append(self.best_params)
        for _ in range(self.parameter_searches):
            self.search_params.append(
                rng.uniform(self.search_boundary[:, 0], self.search_boundary[:, 1]),
            )

    def _find_predicted_minimum(
        self,
        scaled_figure_of_merit_function,
        scaled_search_region,
        params_scaler,
        scaled_jacobian_function=None,
    ):
        """
        Find the predicted minimum of `scaled_figure_of_merit_function()`.

        The search for the minimum is constrained to be within
        `scaled_search_region`.

        The `scaled_figure_of_merit_function()` should take inputs in scaled
        units and generate outputs in scaled units. This is necessary because
        `scipy.optimize.minimize()` (which is used internally here) can struggle
        if the numbers are too small or too large. Using scaled parameters and
        figures of merit brings the numbers closer to ~1, which can improve the
        behavior of `scipy.optimize.minimize()`.

        Args:
            scaled_figure_of_merit_function (function): This should be a
                function which accepts an array of scaled parameter values and
                returns a predicted figure of merit. Importantly, both the input
                parameter values and the returned value should be in scaled
                units.
            scaled_search_region (array): The scaled parameter-space bounds for
                the search. The returned minimum position will be constrained to
                be within this region. The `scaled_search_region` should be a 2D
                array of shape `(self.num_params, 2)` where the first column
                specifies lower bounds and the second column specifies upper
                bounds for each parameter (in scaled units).
            params_scaler (mloop.utilities.ParameterScaler): A `ParameterScaler`
                instance for converting parameters to scaled units.
            scaled_jacobian_function (function, optional): An optional function
                giving the Jacobian of `scaled_figure_of_merit_function()` which
                will be used by `scipy.optimize.minimize()` if provided. As with
                `scaled_figure_of_merit_function()`, the
                `scaled_jacobian_function()` should accept and return values in
                scaled units. If `None` then no Jacobian will be provided to
                `scipy.optimize.minimize()`. Defaults to `None`.

        Returns:
            best_scaled_params (array): The scaled parameter values which
                minimize `scaled_figure_of_merit_function()` within
                `scaled_search_region`. They are provided as a 1D array of
                values in scaled units.
        """
        # Generate the list of starting points for the search.
        self.update_search_params()

        # Search for parameters which minimize the provided
        # scaled_figure_of_merit_function, starting at a few different points in
        # parameter-space. The search for the next parameters will be performed
        # in scaled units because so.minimize() can struggle with very large or
        # very small values.
        best_scaled_cost = float("inf")
        best_scaled_params = None
        for start_params in self.search_params:
            scaled_start_parameters = params_scaler.transform(
                [start_params],
            )
            # Extract 1D array from 2D array.
            scaled_start_parameters = scaled_start_parameters[0]
            result = so.minimize(
                scaled_figure_of_merit_function,
                scaled_start_parameters,
                jac=scaled_jacobian_function,
                bounds=scaled_search_region,
                tol=self.search_precision,
            )
            # Check if these parameters give better predicted results than any
            # others found so far in this search.
            current_best_scaled_cost = result.fun
            curr_best_scaled_params = result.x
            if current_best_scaled_cost < best_scaled_cost:
                best_scaled_cost = current_best_scaled_cost
                best_scaled_params = curr_best_scaled_params

        return best_scaled_params

    def find_next_parameters(self):
        """
        Get the next parameters to test.

        This method searches for the parameters expected to give the minimum
        biased cost, as predicted by the Gaussian process. The biased cost is
        not just the predicted cost, but a weighted sum of the predicted cost
        and the uncertainty in the predicted cost. See
        `self.predict_biased_cost()` for more information.

        This method additionally increments `self.params_count` appropriately.

        Return:
            next_params (array): The next parameter values to try, stored in a
                1D array.
        """
        # Increment the counter and update the bias function.
        self.params_count += 1
        self.update_bias_function()

        # Define the function to minimize when picking the next parameters.
        def scaled_biased_cost_function(scaled_parameters):
            scaled_biased_cost = self.predict_biased_cost(
                scaled_parameters,
                perform_scaling=False,
            )
            return scaled_biased_cost

        # Set bounds on the parameter-space for the search.
        # Here we set search_region to parameters' boundary
        scaled_search_region = self.params_scaler.transform(self.search_boundary.T).T

        # Find the scaled parameters which minimize the biased cost function.
        next_scaled_params = self._find_predicted_minimum(
            scaled_figure_of_merit_function=scaled_biased_cost_function,
            scaled_search_region=scaled_search_region,
            params_scaler=self.params_scaler,
        )

        # Convert the scaled parameters to real/unscaled units.
        next_params = self.params_scaler.inverse_transform([next_scaled_params])[0]

        return next_params
