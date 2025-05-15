import matplotlib.pyplot as plt
import numpy as np
import pytest
import scipy.stats as stats
from scipy.linalg import cho_solve, cholesky

from sklearn.gaussian_process._gpr import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.utils._testing import assert_almost_equal

# Testing Data taken from `test_gpr.py`

GPR_CHOLESKY_LOWER = True
MIN_TANGENT_VAL = 1.0e-6
ALPHA_NOISE = 1.0e-5
SEED = 2025


def f(x):
    return x * np.sin(x)


kernel = RBF(length_scale=1)
length_scales = [0.1, 0.5, 1.0, 2.0, 5.0]

X = np.atleast_2d([1.0, 3.0, 5.0, 6.0, 7.0, 8.0]).T
y = f(X).ravel()

gpr = GaussianProcessRegressor(kernel=kernel, alpha=ALPHA_NOISE, random_state=SEED).fit(
    X, y
)


def logLikelihood(length_scale: float) -> float:
    """Return the GPs log likelihood as calculated by the gpr class"""
    kernel.length_scale = length_scale
    return gpr.log_marginal_likelihood(kernel.theta)


def normalLogLikelihood(length_scale: float) -> float:
    """Return the log likelihood of the multivariate gaussian parametrized by gpr
    prior"""
    kernel.length_scale = length_scale
    cov = kernel(X)
    cov[np.diag_indices_from(cov)] += ALPHA_NOISE
    mean = np.zeros(X.shape[0])
    return stats.multivariate_normal.logpdf(y.flatten(), mean=mean, cov=cov)


def manualTangentCalc(length_scale: float) -> float:
    """Return a close approximation of the derivative of the GP's log likelihood
    using a tangent line with small delta"""
    y1 = logLikelihood(length_scale)
    y2 = logLikelihood(length_scale + MIN_TANGENT_VAL)
    return (y2 - y1) / MIN_TANGENT_VAL


def logLikelihoodDerivCal(length_scale: float) -> float:
    """Return the GPs log likelihood derivative as calculated in the grp
    object"""
    kernel.length_scale = length_scale
    return gpr.log_marginal_likelihood(kernel.theta, eval_gradient=True)[1][0]


def CORRECTED_LogLikelihoodDerivCal(length_scale: float) -> float:
    """Return the correct log likelihood derivative calculation"""
    kernel_ = RBF(length_scale=length_scale)
    y_train = y
    if y_train.ndim == 1:
        y_train = y_train[:, np.newaxis]
    K, K_gradient = kernel_(X, eval_gradient=True)
    ####################################################################################
    # The gradient of the kernel function is made with respect to the log of the
    # hyper parameter (length scale). To correct for this the chain rule shows:
    #
    # df(x)/dln(x) * dln(x)/dx = df(x)/dx
    # df(x)/dln(x) * 1 / x = df(x)/dx
    #
    # Giving correction:

    K_gradient = (
        K_gradient / length_scale
    )  # THIS IS THE ADDED STEP CORRECTING THE LOG LIKELIHOOD
    ####################################################################################

    K[np.diag_indices_from(K)] += ALPHA_NOISE
    L = cholesky(K, lower=GPR_CHOLESKY_LOWER, check_finite=False)
    alpha = cho_solve((L, GPR_CHOLESKY_LOWER), y_train, check_finite=False)

    inner_term = np.einsum("ik,jk->ijk", alpha, alpha)
    K_inv = cho_solve((L, GPR_CHOLESKY_LOWER), np.eye(K.shape[0]), check_finite=False)
    inner_term -= K_inv[..., np.newaxis]

    log_likelihood_gradient_dims = 0.5 * np.einsum(
        "ijl,jik->kl", inner_term, K_gradient
    )

    log_likelihood_gradient = log_likelihood_gradient_dims.sum(axis=-1)

    return log_likelihood_gradient


@pytest.mark.parametrize("length_scale", length_scales)
def test_gpr_log_likelihood(length_scale):
    """Compare the log likelihood given by the gpr against the log likelihood of an
    equivilent multivariate normal distribution"""
    assert_almost_equal(logLikelihood(length_scale), normalLogLikelihood(length_scale))


@pytest.mark.parametrize("length_scale", length_scales)
def test_gpr_derivative_log_likelihood(length_scale):
    """
    ************************** THE PROBLEM - CURRENTLY FAILING *************************

    Compare the log likelihood derivative given by the gpr against the tangent line
    approximationthis test requires `test_gpr_log_likelihood` to pass thus ensuring
    correct tangent approximations

    ************************************************************************************
    """
    assert_almost_equal(
        manualTangentCalc(length_scale), logLikelihoodDerivCal(length_scale)
    )


@pytest.mark.parametrize("length_scale", length_scales)
def test_CORRECTED_gpr_derivative_log_likelihood(length_scale):
    """
    ************************************ CORRECTED *************************************

    Compare the corrected log likelihood derivative of the GP against the tangent line
    approximation this test requires `test_gpr_log_likelihood` to pass thus ensuring
    correct tangent approximations

    ************************************************************************************
    """
    assert_almost_equal(
        manualTangentCalc(length_scale),
        CORRECTED_LogLikelihoodDerivCal(length_scale),
        0.0001,
    )


@pytest.fixture(scope="session", autouse=True)
def saveSupportingPlot():
    yield
    x_ax_start, x_ax_end = 0, 25
    length_scale_axis = np.linspace(x_ax_start, x_ax_end, 1_000)
    log_likelihoods = np.apply_along_axis(
        logLikelihood, axis=1, arr=length_scale_axis.reshape(-1, 1)
    )
    log_likelihoods_deriv_manual = np.apply_along_axis(
        manualTangentCalc, axis=1, arr=length_scale_axis.reshape(-1, 1)
    )
    log_likelihoods_deriv_built_in = np.apply_along_axis(
        logLikelihoodDerivCal, axis=1, arr=length_scale_axis.reshape(-1, 1)
    )
    log_likelihoods_deriv_corrected = np.apply_along_axis(
        CORRECTED_LogLikelihoodDerivCal, axis=1, arr=length_scale_axis.reshape(-1, 1)
    )

    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(12, 5), sharex=True)
    fig.suptitle("Log Likelihood and Its Derivatives", fontsize=16)

    # Plot 1: log_likelihoods with log_likelihoods_deriv_manual
    axes[0].axhline(y=0, color="k", linestyle="-", alpha=0.7)
    axes[0].plot(length_scale_axis, log_likelihoods, "b-", label="Log Likelihood")
    axes[0].plot(
        length_scale_axis,
        log_likelihoods_deriv_manual,
        "r--",
        label="Manual Derivative",
    )
    axes[0].set_xlim(x_ax_start, x_ax_end)
    axes[0].set_ylabel("log likelihood")
    axes[0].set_xlabel("Length Scale")
    axes[0].legend()
    axes[0].set_title("Log Likelihood and Manual Derivative")
    axes[0].grid(True)

    # Plot 2: log_likelihoods with log_likelihoods_deriv_corrected
    axes[1].axhline(y=0, color="k", linestyle="-", alpha=0.7)
    axes[1].plot(length_scale_axis, log_likelihoods, "b-", label="Log Likelihood")
    axes[1].plot(
        length_scale_axis,
        log_likelihoods_deriv_corrected,
        "m--",
        label="Corrected Derivative",
    )
    axes[1].set_xlim(x_ax_start, x_ax_end)
    axes[1].set_ylabel("log likelihood")
    axes[1].set_xlabel("Length Scale")
    axes[1].legend()
    axes[1].set_title("Log Likelihood and Corrected Derivative")
    axes[1].grid(True)

    # Plot 3: log_likelihoods with log_likelihoods_deriv_built_in
    axes[2].axhline(y=0, color="k", linestyle="-", alpha=0.7)
    axes[2].plot(length_scale_axis, log_likelihoods, "b-", label="Log Likelihood")
    axes[2].plot(
        length_scale_axis,
        log_likelihoods_deriv_built_in,
        "g--",
        label="Built-in Derivative",
    )
    axes[2].set_xlim(x_ax_start, x_ax_end)
    axes[2].set_ylabel("log likelihood")
    axes[2].set_xlabel("Length Scale")
    axes[2].legend()
    axes[2].set_title("Log Likelihood and Built-in Derivative")
    axes[2].grid(True)

    # Adjust layout
    plt.tight_layout()
    plt.show()

    plt.savefig("log_likelihood_comparison.png")
