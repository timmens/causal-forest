#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
Module to fit a causal forest.

This module uses the causal tree algorithm from `algorithm.causaltree.py` to
fit a causal forest on given data.
"""
import glob
import warnings
from itertools import count

import numpy as np
import pandas as pd

from causal_forest.tree.causaltree import fit_causaltree
from causal_forest.tree.causaltree import predict_causaltree


class CausalForest:
    """Estimator class to fit a causal forest.

    Estimator class to fit a causal forest on numerical data given
    hyperparameters set in *forest_params* and *tree_params*.
    Provides methods to `fit` the model, `predict` using the fitted model,
    `save` the fitted model and `load` a fitted model.

    Note that the structure of this estimator is based on the BaseEstimator and
    RegressorMixin from sklearn; however, here we predict treatment effects
    --which are unobservable-- hence regular model validation and model
    selection techniques (e.g. cross validation grid search) do not work as
    we can never estimate a loss on a training sample.


    Attributes:
        forest_params (dict):
        tree_params (dict):

    """

    def __init__(self, forest_params=None, tree_params=None, seed_counter=1):
        """Initiliazes CausalForest estimator with hyperparameters.

        Initializes CausalForest estimator with hyperparameters for the forest,
        i.e. *forest_params*, which contains the number of trees that should
        be fit and the ratio of features to be randomly considered at each
        split; and for the trees of which the forest is made of we consider
        *tree_params*, which contains the minimum number of observations in
        a leaf node and the maximum depth of a tree.

        Args:
            forest_params (dict): Hyperparameters for forest. Has to include
                'num_trees' (int) and 'ratio_features_at_split' (in [0, 1]).
            tree_params (dict): Parameters for tree. Has to include 'min_leaf'
                (int) and 'max_depth' (int).
        """
        self._is_fitted = False
        self._num_features = None
        self.fitted_model = None
        self.seed_counter = seed_counter

        if forest_params is None:
            self.forest_params = {
                "num_trees": 100,
                "ratio_features_at_split": 0.5,
            }
        else:
            if not isinstance(forest_params, dict):
                raise TypeError(
                    "Argument *forest_params* is not a dictionary."
                )

            if {"num_trees", "ratio_features_at_split"} != set(forest_params):
                raise ValueError(
                    "Argument *forst_params* does not contain the correct "
                    "parameter 'num_trees' and 'ratio_features_at_split'."
                )

            self.forest_params = forest_params

        if tree_params is None:
            self.tree_params = {
                "min_leaf": 5,
                "max_depth": 25,
            }
        else:
            if not isinstance(tree_params, dict):
                raise TypeError("Argument *tree_params* is not a dictionary.")

            if {"min_leaf", "max_depth"} != set(tree_params):
                raise ValueError(
                    "Argument *tree_params* does not contain the correct "
                    "parameter 'min_leaf' and 'max_depth'."
                )

            self.tree_params = tree_params

    def fit(self, X, t, y):
        """Fits causal forest on supplied data.

        Fits a causal forest on outcomes *y* with treatment status *t* and
        features *X*, if data has no missing values and is of consistent shape.

        Args:
            X: data on features
            t: data on treatment status
            y: data on outcomes

        Returns:
            self: the fitted regressor.

        Raises:
            - TypeError, if data is not a pd.DataFrame or np.array
            - ValueError, if data has inconsistent shapes

        """
        _assert_data_input_causal_forest(X, t, y)

        fitted_model = fit_causalforest(
            X=X,
            t=t,
            y=y,
            forest_params=self.forest_params,
            tree_params=self.tree_params,
            seed_counter=self.seed_counter,
        )
        self.fitted_model = fitted_model
        self._is_fitted = True
        self._num_features = X.shape[1]

        return self

    def predict(self, X):
        """Predicts treatment effects for new features *X*.

        If the regressor has been fitted, predicts treatment effects of new
        features *X*, if *X* is a np.array of pd.DataFrame of correct shape.

        Args:
            X: data on new features

        Returns:
            predictions (np.array): array containing one predictions per row
                of *X*.

        """
        if not self._is_fitted:
            warnings.warn(
                "The CausalForest has not yet been fitted; hence the predict "
                "method cannot be called.",
                UserWarning,
            )
            return None
        if not isinstance(X, np.ndarray) and not isinstance(X, pd.DataFrame):
            raise TypeError(
                "Data on new features *X* is not a pd.DataFrame or np.array."
            )
        if X.shape[1] != self._num_features:
            raise ValueError("Data on new features *X* has wrong dimensions.")

        predictions = predict_causalforest(self.fitted_model, X)
        return predictions

    def save(self, filename, overwrite=True):
        """Save fitted model as a csv file.

        Args:
            filename (str): complete directory path including filename where to
                save the fitted model.
            overwrite (bool): overwrite existing file if True and not
                otherwise

        Returns:
            None

        """
        if not self._is_fitted:
            warnings.warn(
                "The CausalForest has not yet been fitted; hence it cannot be "
                "saved.",
                UserWarning,
            )

        if not overwrite:
            file_exists = glob.glob(filename)
            if file_exists:
                warnings.warn("File already exists.", UserWarning)
                return None

        self.fitted_model.to_csv(filename)
        return None

    def load(self, filename, overwrite_fitted_model=False):
        """Load fitted model from disc.

        Args:
            filename (str): complete directory path including filename where to
                load the fitted model from.
            overwrite_fitted_model (bool): overwrite self.fitted_model if True
                and do nothing otherwise.

        Returns:
            None

        """
        if self._is_fitted and not overwrite_fitted_model:
            warnings.warn(
                "Cannot load model as CausalForest object is already fitted "
                "and *overwrite_fitted_model* is set to false.",
                UserWarning,
            )
            return None

        candidate_model = pd.read_csv(filename)
        try:
            candidate_model = candidate_model.set_index(["tree_id", "node_id"])
        except KeyError:
            raise KeyError(
                "The file to load needs to have its first two"
                "columns to be 'tree_id' and 'node_id'"
            )
        _assert_df_is_valid_causal_forest(candidate_model)

        fitted_model = _update_dtypes(candidate_model)
        self.fitted_model = fitted_model
        self._is_fitted = True
        return self


def fit_causalforest(X, t, y, forest_params, tree_params, seed_counter=1000):
    """Fits a causal forest on given data.

    Fits a causal forest using data on outcomes *y*, treatment status *t*
    and features *X*. Forest will be made up of *num_trees* causal trees with
    parameters *crit_params_ctree*.

    Args:
        X (np.array): 2d array containing numerical features
        t (np.array): 1d (boolean) array containing treatment status
        y (np.array): 1d array containing outcomes
        forest_params (dict): dictionary containing hyperparameters for forest
        tree_params (dict): dictionary containing parameters for tree
        seed_counter (int): number where to start the seed counter

    Returns:
        cforest (pd.DataFrame): fitted causal forest represented in a pandas
            data frame.

    """
    n = len(y)
    counter = count(seed_counter)
    num_trees = forest_params["num_trees"]
    ratio_features_at_split = forest_params["ratio_features_at_split"]

    cforest = []
    for _i in range(num_trees):
        seed = next(counter)
        resample_index = _draw_resample_index(n, seed)
        ctree = fit_causaltree(
            y[resample_index],
            t[resample_index],
            X[resample_index],
            tree_params,
        )
        cforest.append(ctree)

    cforest = pd.concat(
        cforest, keys=range(num_trees), names=["tree_id", "node_id"]
    )

    return cforest


def _draw_resample_index(n, seed):
    """Compute vector of randomly drawn indices with replacement.

    Draw indices with replacement from the discrete uniform distribution
    on {0,...,n-1}. We control the randomness by setting the seed to *seed*.
    If *seed* = -1 we return all indices {0,1,2,...,n-1} for debugging.

    Args:
        n (int): Upper bound for indices and number of indices to draw
        seed (int): Random number seed.

    Returns:
        indices (np.array): Resample indices (resampled with replacement)

    """
    if seed == -1:
        return np.arange(n)
    np.random.seed(seed)
    indices = np.random.randint(0, n, n)
    return indices


def predict_causalforest(cforest, X):
    """Predicts individual treatment effects for a causal forest.

    Predicts individual treatment effects for new observed features *X*
    on a fitted causal forest *cforest*.

    Args:
        cforest (pd.DataFrame): fitted causal forest represented in a multi-
            index pd.DataFrame consisting of several fitted causal trees
        X (np.array): 2d array of new observations for which we predict the
            individual treatment effect.

    Returns:
        predictions (np.array): 1d array of treatment predictions.

    """
    num_trees = len(cforest.groupby(level=0))
    n, p = X.shape

    predictions = np.empty((num_trees, n))
    for i in range(num_trees):
        predictions[i, :] = predict_causaltree(cforest.loc[i], X)

    predictions = predictions.mean(axis=0)
    return predictions


def _assert_data_input_causal_forest(X, t, y):
    """Assert if input data satisfies restrictions.

    Asserts if input data is either a pd.DataFrame/pd.Sereis or np.array and if
    data has consistent dimensions and values, i.e. X is a numerical 2 dim.
    matrix, y is a numerical 1 dim. vector and t is a boolean 1 dim. vector,
    and all data has no missing valus.

    Args:
        X: data on features
        t: data on treatment status
        y: data on outcomes

    Returns: True if data satisfies all restrictions and raises Error otherwise

    Raises:
        - TypeError, if data is not a pd.DataFrame/pd.Series or np.array
        - ValueError, if data has inconsistent shapes

    """
    # assert dtype of features *X*
    if isinstance(X, np.ndarray):
        if np.isnan(np.sum(X)):
            raise ValueError("Data on features *X* contains NaNs.")
        nx, p = X.shape
    elif isinstance(X, pd.DataFrame):
        if X.isnull().any().any():
            raise ValueError("Data on features *X* contains NaNs.")
        nx, p = X.shape
    else:
        raise TypeError(
            "Data on features *X* is not a pd.DataFrame or np.array."
        )

    # assert dtype of treatment status *t*
    if isinstance(t, np.ndarray):
        if np.isnan(np.sum(t)):
            raise ValueError("Data on treatment status *t* contains NaNs.")
        else:
            if t.dtype.kind != "b":
                raise ValueError(
                    "Data on treatment status *t* is not boolean."
                )
            nt = len(t)
    elif isinstance(t, pd.Series):
        if t.isnull().any():
            raise ValueError("Data on treatment status *t* contains NaNs.")
        else:
            if t.dtype.kind != "b":
                raise ValueError(
                    "Data on treatment status *t* is not boolean."
                )
            nt = len(t)
    else:
        raise TypeError(
            "Data on treatment status *t* is not a pd.Series or np.array."
        )

    # assert dtype of outcomes *y*
    if isinstance(y, np.ndarray):
        if np.isnan(np.sum(y)):
            raise ValueError("Data on outcomes *y* contains NaNs.")
        ny = len(y)
    elif isinstance(y, pd.Series):
        if y.isnull().any():
            raise ValueError("Data on outcomes *y* contains NaNs.")
        ny = len(y)
    else:
        raise TypeError("Data on outcomes *y* is not a pd.Series or np.array.")

    if not nx == nt == ny:
        raise ValueError("Dimensions of dat   a is not consistent.")

    return True


def _assert_df_is_valid_causal_forest(candidate_model):
    """Assert *df* represents valid causal forest.

    A valid causal forest model is given by a pd.DataFrame which fulfills the
    following criteria:
    1 (MultiIndex). The data frame *df* must have a MultiIndex with the
    first layer 'tree_id' and the second layer 'node_id'.
    2 (Column names). The column names of *df* must match exactly with
    ["left_child", "right_child", "level", "split_feat", "split_value",
    "treat_effect"]
    3 (Column dtype).
    left_child: int
    right_child: int
    level: int
    split_feat: int
    split_value: float
    treat_effect: float

    Args:
        candidate_model (pd.DataFrame): a candidate fitted model

    Returns: True if *df* constitutes a valid causal forest and raises Error
        otherwise

    Raises:
        ValueError, if *df* does represent a valid causal forest model.

    """
    # MultiIndex
    if not isinstance(candidate_model.index, pd.MultiIndex):
        raise ValueError(
            "Candidate model does not represent a valid causal forest as the "
            "index is not of type pd.MultiIndex."
        )
    else:
        if ["tree_id", "node_id"] != candidate_model.index.names:
            raise ValueError(
                "Candidate model does not represent a valid causal forest as "
                "names of index are not 'tree_id' and 'node_id'."
            )

    # Column names
    column_names = [
        "left_child",
        "right_child",
        "level",
        "split_feat",
        "split_value",
        "treat_effect",
    ]
    if set(column_names) != set(candidate_model.columns):
        raise ValueError(
            "Candidate model does not represent a valid causal forest as the "
            "set of column names is not equal to {'left_child', 'right_child',"
            "'level', 'split_feat', 'split_value', 'treat_effect'}"
        )

    # Column data types
    int_columns = column_names[:4]
    for int_col in int_columns:
        _is_int = (
            candidate_model[int_col]
            .apply(
                lambda x: True if np.isnan(x) else float.is_integer(float(x))
            )
            .all()
        )
        if not _is_int:
            raise ValueError(f"Data type of column {int_col} is not int.")

    return True


def _update_dtypes(candidate_model):
    """Update dtypes of specific columns candidate_model from float to int.

    Updates dtypes of candidate_model so that columns representing integer
    are set to integers.

    Args:
        candidate_model (pd.DataFrame):

    Returns:
        fitted_model (pd.DataFrame): updated model

    """
    columns_to_int = [
        "left_child",
        "right_child",
        "level",
        "split_feat",
    ]

    fitted_model = candidate_model.copy()
    fitted_model[columns_to_int] = fitted_model[columns_to_int].astype("Int64")

    return fitted_model
