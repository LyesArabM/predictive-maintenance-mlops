import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

AIR = "Air temperature [K]"
PROC = "Process temperature [K]"
RPM = "Rotational speed [rpm]"
TORQUE = "Torque [Nm]"
WEAR = "Tool wear [min]"

DERIVED_NAMES = ["temp_diff", "power", "strain"]


class DerivedFeatures(BaseEstimator, TransformerMixin):
    """Adds the three quantities used by the failure rules.

    temp_diff = process temp - air temp   (heat dissipation rule)
    power     = torque * angular speed    (power rule, in W)
    strain    = tool wear * torque        (overstrain rule)
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        X["temp_diff"] = X[PROC] - X[AIR]
        X["power"] = X[TORQUE] * X[RPM] * 2 * np.pi / 60
        X["strain"] = X[WEAR] * X[TORQUE]
        return X