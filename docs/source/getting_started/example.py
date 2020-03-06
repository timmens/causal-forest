"""Helper functions for the example hosted at causal-forest.readthedocs.io."""
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D


def simulate(
    nobs,
    nfeatures,
    coefficients,
    error_var,
    seed,
    propensity_score=None,
    alpha=0.8,
):
    """Simulate data with heterogenous treatment effects.

    Simulate outcomes y (length *nobs*), features X (*nobs* x *nfeatures*) and
    treatment status treatment_status (length *nobs*) using the potential
    outcome framework from Neyman and Rubin.
        We simulate the data by imposing a linear model with two relevant
    features plus a treatment effect. However, we return a design matrix with
    *nfeatures* from which only the two used in the simulation are relevant.
        The model looks the following:
    y(0)_i = coef[0] + X_1i coef[1] +  + X_2i coef[2] + error_i,
    y(1)_i = coef[0] + X_1i coef[1] +  + X_2i coef[2]  + treatment_i + error_i,
    where coef[0], coef[1] and coef[2] denote the intercept and slopes
    respectively, and treatment_i = treatment(X_i) denotes the heterogenous
    treatment effect, which is solely dependent on the location of the
    individual in the feature space of the first two dimensions, i.e. as with
    the linear model it only depends on X_1i and X_2i; at last y(0)_i, y(1)_i
    denote the potential outcomes of individual i.

    Args:
        nobs (int): positive integer denoting the number of observations.
        nfeatures (int): positiv integer denoting the number of features. Must
            be greater than or equal to 2.
        coefficients: Coefficients for the linear model, first value denotes
            the intercept, second and third the slope for the second and third
            feature. Must be convertable to a np.ndarray.
        error_var (float): positive float denoting the error variance.
        seed (int): seed for the random number generator.
        propensity_score (np.array): array containing propensity scores, must
            be of length *nobs*. If None, will be set to 0.5 for each
            individual.
        alpha (float): positive parameter influencing the shape of the
            function. Default is 0.8.

    Returns:
        X (np.array): [nobs x nfeatures] numpy array with simulated features.
        t (np.array): [nobs] boolean numpy array containing treatment
            status of individuals.
        y (np.array): [nobs] numpy array containing "observed" outcomes.

    Raises:
        ValueError, if dimensions mismatch or data type of inputs is incorrect.

    """
    np.random.seed(seed)

    # Assert input values
    if not float.is_integer(float(nobs)):
        raise ValueError("Argument *nobs* is not an integer.")
    if not float.is_integer(float(nfeatures)):
        raise ValueError("Argument *nfeatures* is not an integer.")

    coefficients = np.array(coefficients)
    if nfeatures < 2:
        raise ValueError(
            "Argument *nfeatures* must be greater or equal than" "two"
        )

    if len(coefficients) != 3:
        raise ValueError("Argument *coefficients* needs to be of length 3.")

    # Simulate treatment status
    if propensity_score is None:
        treatment_status = np.random.binomial(1, 0.5, nobs)
    else:
        if len(propensity_score) != nobs:
            raise ValueError(
                "Dimensions of argument *propensity_score* do not"
                "match with *nobs*."
            )
        treatment_status = np.random.binomial(1, propensity_score, nobs)

    error = np.random.normal(0, np.sqrt(error_var), nobs)
    X = np.random.uniform(-15, 15, (nobs, nfeatures))
    y = _compute_outcome(X, coefficients, treatment_status, error, alpha)
    t = np.array(treatment_status, dtype=bool)

    return X, t, y


def _compute_outcome(X, coefficients, treatment_status, error, alpha):
    """Compute observed potential outcome.

    Simulates potential outcomes and returns outcome corresponding to the
    treatment status.

    Args:
        X (np.array): design array containing features.
        coefficients (np.array): coefficient array containing intercept and
            two slope parameter.
        treatment_status (np.array): array containing the treatment status
        error (np.array): array containing the error terms for the linear model
        alpha (float): positive parameter influencing the shape of the function

    Returns:
        y (np.array): the observed potential outcomes.

    """
    baseline_model = coefficients[0] + np.dot(X[:, :2], coefficients[1:])
    y0 = baseline_model + error

    treat_effect = true_treatment_effect(X[:, 0], X[:, 1], alpha)
    y1 = y0 + treat_effect

    y = (1 - treatment_status) * y0 + treatment_status * y1
    return y


def true_treatment_effect(x, y, alpha=0.8, scale=5):
    """Compute individual treatment effects.

    Computes individual treatment effect conditional on features *X* using
    parameter *alpha* to determine the smoothness of the conditional
    treatment function and *scale* to determine the scaling.

    Args:
        x (np.array): Data on first dimension.
        y (np.array): Data on second dimension.
        alpha (float): Positive parameter influencing the shape of the
            function. Defaults to 0.8.
        scale (float): Positive parameter determining the scaling of the
            function. Defaults to 5. With scale=x the range of the function is
            [0, x].

    Returns:
        result (np.array): Treatment effects.

    """
    denominatorx = 1 + np.exp(-alpha * (x - 1 / 3))
    fractionx = 1 / denominatorx

    denominatory = 1 + np.exp(-alpha * (y - 1 / 3))
    fractiony = 1 / denominatory

    result = scale * fractionx * fractiony
    return result


def plot_treatment_effect(alpha, scale, figsize):
    """Plot the true conditional treatment effect.

    Args:
        alpha (float): Positive parameter influencing the shape of the
            function.
        scale (float): Positive parameter determining the scaling of the
            function.
        figsize (tuple): Figure size.

    Returns:
        ax (matplotlib.axis): The finished plot.

    """
    X, Y = _construct_meshgrid()
    Z = _true_treatment_effect_on_meshgrid(X, Y, alpha=alpha, scale=scale)
    ax = plot_3d_func(X, Y, Z, "True Treatment Effect", figsize)
    return ax


def plot_predicted_treatment_effect(cf, figsize, npoints, num_workers):
    """Plot the predicted treatment effect from a Causal Forest.

    Args:
        cf (CausalForest): Fitted Causal Forest.
        figsize (tuple): The figure size.
        npoints (int): Number of points for meshgrid.
        num_workers (int): Number of workers for parallelization.

    Returns:
        ax (matplotlib.axis): The finished plot.

    """
    X, Y = _construct_meshgrid(npoints=npoints)
    Z = _predicted_treatment_effect_on_meshgrid(X, Y, cf, num_workers)
    ax = plot_3d_func(X, Y, Z, "Predicted Treatment Effect", figsize)
    return ax


def plot_3d_func(X, Y, Z, zlabel, figsize):
    """Plot a 3 dimensional function.

    Plots a 3 dimensional function, where X, Y, Z form a meshgrid, with the
    usual functional relationship: z = f(x, y).

    Args:
        X (np.array): Meshgrid on first dimension.
        Y (np.array): Meshgrid on second dimension.
        Z (np.array): Meshgrid on outcome dimensions.
        zlabel (str): Name of z-axis.
        figsize (tuple): Figure size.

    Returns:
        ax (matplotlib.axis): The finished plot.

    """
    mpl.rcParams.update({"font.family": "stix"})
    mpl.rcParams.update({"font.size": 30})

    plt.rcParams.update({"font.size": 22})

    fig = plt.figure()
    ax = fig.gca(projection=Axes3D.name)
    ax.plot_surface(
        X, Y, Z, rstride=1, cstride=1, cmap="viridis", edgecolor="none"
    )

    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_zlabel(zlabel)
    ax.set_zlim((0, 5))
    ax.yaxis.labelpad = 30
    ax.zaxis.labelpad = 10
    ax.xaxis.labelpad = 30
    ax.view_init(30, 240)

    ax.grid(False)
    ax.xaxis.pane.set_edgecolor("black")
    ax.yaxis.pane.set_edgecolor("black")
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    plt.rcParams["figure.figsize"] = [figsize[0], figsize[1]]


def _construct_meshgrid(left=-15, right=15, npoints=100):
    x = np.linspace(left, right, npoints)
    y = x.copy()
    X, Y = np.meshgrid(x, y)
    return X, Y


def _meshgrid_to_1darray(X, Y):
    n = len(X)
    XX = X.reshape((n, n, 1))
    YY = Y.reshape((n, n, 1))
    XY = np.concatenate((XX, YY), axis=2)
    return XY.reshape((-1, 2))


def _true_treatment_effect_on_meshgrid(X, Y, alpha, scale):
    """Compute true treatment effect on meshgrid.

    Args:
        X (np.array): Meshgrid on first dimension.
        Y (np.array): Meshgrid on second dimension.
        alpha (float): positive parameter influencing the shape of the
            function.
        scale (float): positive parameter determining the scaling of the
            function.

    Returns:
        out (np.array): Meshgrid on true treatment effect.

    """
    Z = true_treatment_effect(X, Y, alpha=alpha, scale=scale)
    return Z


def _predicted_treatment_effect_on_meshgrid(X, Y, cf, num_workers):
    """Compute predicted treatment effect on meshgrid.

    Compute predicted treatment effect of Causal Forest *cf* on the first
    two input dimensions.

    Args:
        X (np.array): Meshgrid on first dimension.
        Y (np.array): Meshgrid on second dimension.
        cf (CausalForest): Fitted CausalForest object.
        num_workers (int): Number of workers for parallelization.

    Returns:
        out (np.array): Meshgrid on predicted treatment effect.

    """
    XY = _meshgrid_to_1darray(X, Y)

    k = cf.num_features
    n = len(XY)
    fill = np.zeros((n, k - 2))
    XY = np.concatenate((XY, fill), axis=1)

    out = cf.predict(XY, num_workers)
    return out.reshape(X.shape)
