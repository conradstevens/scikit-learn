"""Script to generate plots for README.md"""

# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause


import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import make_interp_spline
from scipy.stats import norm, t

from sklearn.gaussian_process import GaussianProcessRegressor, TProcessRegressor
from sklearn.gaussian_process.kernels import (
    RBF,
    ConstantKernel,
    ExpSineSquared,
)


def norm_conf_gp(std):
    """Returns the 95% confidence interval of the normal distribution"""
    zScore = norm(0, 1).ppf(0.975)
    confidence = zScore * np.sqrt(std)
    return confidence


def norm_conf_tp(std, df):
    """Returns the 95% confidence interval of the normal distribution"""
    tScore = t.ppf(0.975, df)
    confidence = tScore * std
    return confidence


vec_norm_conf_gp = np.vectorize(norm_conf_gp)
vec_norm_conf_tp = np.vectorize(norm_conf_tp)


def target_generator(X, add_noise=False):
    """Generates function to plot"""
    target = 0.5 * np.sin(X)
    if add_noise:
        rng = np.random.RandomState(1)
        target += rng.normal(0, 0.3, size=target.shape)
    return target.squeeze()


def plot_sin_comparison():
    """Plot GP and TP comparison"""
    c, l, p, l2 = 1, 2, 6, 8
    kernel = (
        ConstantKernel(constant_value=c, constant_value_bounds="fixed")
        * RBF(length_scale=l2, length_scale_bounds="fixed")
        * ExpSineSquared(
            length_scale=l,
            periodicity=p,
            length_scale_bounds="fixed",
            periodicity_bounds="fixed",
        )
    )

    n = 40
    X = np.linspace(-30, 30, num=500).reshape(-1, 1)
    y = target_generator(X, add_noise=False)
    X_train = np.linspace(-10, 10, num=n) + 0.1 * np.array(
        [norm.rvs() for _ in range(n)]
    )
    X_train = X_train.reshape(-1, 1)
    y_train = target_generator(X_train, add_noise=False)

    gpr = GaussianProcessRegressor(kernel=kernel, alpha=0.0001).fit(
        X_train, y_train
    )  # , optimizer=None)
    tpr = TProcessRegressor(kernel=kernel, v=3, alpha=0.0001).fit(
        X_train, y_train
    )  # , optimizer=None)

    ### Plot GP ###
    gp_mean, gp_std = gpr.predict(X, return_std=True)
    confs = vec_norm_conf_gp(gp_std)
    topConf = gp_mean + confs
    botConf = gp_mean - confs

    plt.xlabel("X")
    _ = plt.ylabel("y")

    plt.fill_between(np.reshape(X, -1), botConf, topConf, color="blue", alpha=0.1)
    plt.plot(X, y, label="Ground Truth", color="red", linewidth=1.2)
    plt.plot(X, gp_mean, label="GP", color="b", linewidth=1.2)
    plt.scatter(
        x=X_train[:, 0],
        y=y_train,
        color="b",
        alpha=1,
        marker="x",
        s=20,
        label="Observations",
    )

    plt.legend()
    plt.ylim(-3, 3)
    plt.show()

    ### Plot TP ###
    tp_mean, tp_std = tpr.predict(X, return_std=True)
    confs = vec_norm_conf_tp(tp_std, tpr.v)
    topConf = tp_mean + confs
    botConf = tp_mean - confs

    plt.xlabel("X")
    _ = plt.ylabel("y")

    plt.fill_between(np.reshape(X, -1), botConf, topConf, color="blue", alpha=0.1)
    plt.plot(X, y, label="Ground Truth", color="red", linewidth=1.2)
    plt.plot(X, tp_mean, label="TP", color="b", linewidth=1.2)
    plt.scatter(
        x=X_train[:, 0],
        y=y_train,
        color="b",
        alpha=1,
        marker="x",
        s=20,
        label="Observations",
    )

    plt.legend()
    plt.ylim(-3, 3)
    plt.show()


def plot_rbf_samples():
    """Plot sampled functions form GP and TP"""
    kernel = RBF(length_scale=1, length_scale_bounds="fixed")
    X = np.linspace(-30, 30, 120).reshape(-1, 1)

    ### GPs ###
    gp = GaussianProcessRegressor(kernel=kernel, alpha=0.0001)
    gp_mean, gp_std = gp.predict(X, return_std=True)
    confs = vec_norm_conf_gp(gp_std)
    gp_topConf = gp_mean + confs
    gp_botConf = gp_mean - confs

    plt.plot(X, gp_mean, label="GP Prior", color="b", linewidth=1.2, alpha=1)
    plt.fill_between(X.flatten(), gp_botConf, gp_topConf, color="blue", alpha=0.1)

    for i in range(1, 40):
        y_samp = gp.sample_y(X, random_state=i)

        # Apply cubic smoothener
        spl = make_interp_spline(X.flatten(), y_samp.flatten(), k=3)
        x_smooth = np.linspace(-30, 30, 500)
        y_smooth = spl(np.linspace(-30, 30, 500))
        if i == 1:
            plt.plot(
                x_smooth,
                y_smooth,
                label="sampled GP",
                color="b",
                linewidth=1.2,
                alpha=0.3,
            )
        else:
            plt.plot(x_smooth, y_smooth, color="b", linewidth=1.2, alpha=0.3)

    plt.ylim(-7, 7)
    plt.legend()
    plt.show()

    ### TPs ###
    tp = TProcessRegressor(kernel=kernel, alpha=0.0001)

    tp_mean, tp_std = gp.predict(X, return_std=True)
    confs = vec_norm_conf_gp(tp_std)
    tp_topConf = tp_mean + confs
    tp_botConf = tp_mean - confs

    plt.plot(X, gp_mean, label="TP Prior", color="b", linewidth=1.2, alpha=1)
    plt.fill_between(X.flatten(), tp_botConf, tp_topConf, color="blue", alpha=0.1)

    for i in range(1, 40):
        y_samp = tp.sample_y(X, random_state=i)

        # Apply cubic smoothener
        spl = make_interp_spline(X.flatten(), y_samp.flatten(), k=3)
        x_smooth = np.linspace(-30, 30, 500)
        y_smooth = spl(np.linspace(-30, 30, 500))
        if i == 1:
            plt.plot(
                x_smooth,
                y_smooth,
                label="labeled TP",
                color="b",
                linewidth=1.2,
                alpha=0.3,
            )
        else:
            plt.plot(x_smooth, y_smooth, color="b", linewidth=1.2, alpha=0.3)

    plt.ylim(-7, 7)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    plot_sin_comparison()
    plot_rbf_samples()
