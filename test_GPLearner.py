# %%
from learners import GaussianProcessLearner
import numpy as np
import matplotlib.pyplot as plt


# %%
def rastrigin(params):
    """
    Rastrigin function for 2D optimization.

    Args:
      x: A float representing the x-coordinate.
      y: A float representing the y-coordinate.

    Returns:
      The value of the Rastrigin function at the given (x, y) point.
    """
    x, y = params
    A = 10
    return A * 2 + x**2 - A * np.cos(5 * np.pi * x) + y**2 - A * np.cos(5 * np.pi * y)


# %%
gpl = GaussianProcessLearner(num_params=2)
# %%
print(
    "Try scaling: \n",
    gpl.params_scaler.transform(
        np.random.rand(100 * gpl.num_params).reshape((100, gpl.num_params))
    ),
)
print("Check scaler: ", gpl.params_scaler.scale_, gpl.params_scaler.data_min_)
# %%
N = 150
# cost_func = lambda params: -np.sum(np.sinc(params * 5))
cost_func = rastrigin
params = (np.random.rand(N * 2).reshape((N, 2)) - 0.5) * 2
costs = np.asarray(list(map(cost_func, params)))

for i in range(N):
    gpl._update_run_data_attributes(params[i], costs[i], 0, i)

plt.scatter(params[:, 0], params[:, 1], c=costs)
plt.show()

# %%
gpl.fit_gaussian_process()

# %%  Plot observe with random
M = 100
x = y = np.linspace(-1, 1, M)
X = np.zeros((M, M, 2))
for i in range(M):
    for j in range(M):
        X[i, j] = [x[i], y[j]]

Y = np.asarray(list(map(cost_func, X.reshape(-1, 2))))
Y_mean, Y_var = gpl.predict_cost(X.reshape(-1, 2), return_uncertainty=True)
gpl.params_count = 3
gpl.update_bias_function()
Biased_cost = gpl.predict_biased_cost(X.reshape(-1, 2))

Y = Y.reshape(M, M)
Y_var = Y_var.reshape(M, M)
Y_mean = Y_mean.reshape(M, M)
Biased_cost = Biased_cost.reshape(M, M)

# plt.plot(X[50, :, 1], Y[50, :])
# plt.plot(X[50, :, 1], Y_hat[50, :])
plt.scatter(X[:, :, 0], X[:, :, 1], c=Y_mean)
plt.show()
plt.scatter(X[:, :, 0], X[:, :, 1], c=Y_var)
plt.show()
plt.scatter(X[:, :, 0], X[:, :, 1], c=Biased_cost)
plt.show()

# %% Test searching params
gpl = GaussianProcessLearner(num_params=2)

param = (np.random.rand(2) - 0.5) * 2
cost = cost_func(param)

gpl._update_run_data_attributes(param, cost, 0, 0)
gpl.fit_gaussian_process()

for i in range(1, N):
    # gpl.params_count += 1
    # gpl.update_bias_function()
    # Y_bias = gpl.predict_biased_cost(X.reshape(-1, 2)).reshape(M, M)
    # gpl.params_count -= 1

    next_param = gpl.find_next_parameters()
    next_cost = cost_func(next_param)

    gpl._update_run_data_attributes(next_param, next_cost, 0, i)
    gpl.fit_gaussian_process()

    # Y_mean = gpl.predict_cost(X.reshape(-1, 2)).reshape(M, M)
    # plt.scatter(X[:, :, 0], X[:, :, 1], c=Y_mean)
    # plt.scatter(gpl.all_params[-1, 0], gpl.all_params[-1, 1], c="r")
    # plt.show()

Y_mean = gpl.predict_cost(X.reshape(-1, 2)).reshape(M, M)
plt.scatter(gpl.all_params[:, 0], gpl.all_params[:, 1], c=gpl.all_costs)
plt.show()
plt.scatter(X[:, :, 0], X[:, :, 1], c=Y_mean)
plt.show()
