from sklearn.base import BaseEstimator


class FirstOutputWrapper(BaseEstimator):
    """Extract only the first output column from a multi-output model.

    Used by StackingEnsemble and conformal calibration, which need a
    single-output estimator. Defined here (not in train_model.__main__)
    so pickled objects can be loaded in gunicorn without AttributeError.
    """
    def __init__(self, model): self.model = model
    def fit(self, X, y): return self
    def predict(self, X): return self.model.predict(X)[:, 0]
