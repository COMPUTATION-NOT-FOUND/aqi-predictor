"""
Pickle-safe model wrappers.

Every custom estimator class that can end up inside a saved/champion model MUST
live here (an importable module), NOT inside train_model.py. When train_model.py is
run as a script its module name is `__main__`, so a class defined there pickles as
`__main__.WeightedVoter` and cannot be un-pickled by predict.py / the dashboard
(`AttributeError: Can't get attribute 'WeightedVoter'`). Defining them here pins the
module path so any process that imports this module can load the artifact.
"""
import numpy as np
from sklearn.base import BaseEstimator


class FirstOutputWrapper(BaseEstimator):
    """Extract only the first output column from a multi-output model.

    Used by StackingEnsemble and conformal calibration, which need a
    single-output estimator.
    """
    def __init__(self, model): self.model = model
    def fit(self, X, y): return self
    def predict(self, X): return self.model.predict(X)[:, 0]


class WeightedVoter:
    """Weighted average of several multi-output regressors (VotingEnsemble)."""
    def __init__(self, models, weights):
        self.models  = models
        self.weights = weights

    def fit(self, X, Y):
        for m in self.models:
            m.fit(X, Y)
        return self

    def predict(self, X):
        preds = np.stack([m.predict(X) for m in self.models], axis=0)
        return np.einsum("i,ijk->jk", self.weights, preds)


class KerasWrapper:
    """Adapt a Keras model to the sklearn-style .predict(X) interface (DistilledMLP)."""
    def __init__(self, model): self.model = model

    def predict(self, X):
        return self.model.predict(X, verbose=0)
