"""Convert columns to appropriate, memory-efficient dtypes."""

from __future__ import annotations
import pandas as pd
import numpy as np

from ._utils import is_string_like


class DTypeConverter:
    """
    Optimizes column dtypes:
        - Numeric columns -> smallest safe int/float subtype
        - Object columns with low cardinality -> 'category'
        - Boolean-like object columns -> bool
    """

    BOOL_TRUE = {"true", "t", "yes", "y", "1"}
    BOOL_FALSE = {"false", "f", "no", "n", "0"}

    def __init__(
        self,
        downcast_numerics: bool = True,
        infer_categoricals: bool = True,
        category_threshold: float = 0.5,
        infer_booleans: bool = True,
    ):
        """
        Args:
            category_threshold: Convert to category if
                (unique values / total values) <= threshold.
        """
        self.downcast_numerics = downcast_numerics
        self.infer_categoricals = infer_categoricals
        self.category_threshold = category_threshold
        self.infer_booleans = infer_booleans
        self.actions_: list[str] = []

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        self.actions_ = []
        before_mem = df.memory_usage(deep=True).sum()

        for col in df.columns:
            old_dtype = df[col].dtype

            # Boolean inference on string-like columns
            if self.infer_booleans and is_string_like(df[col]):
                bool_series = self._try_to_bool(df[col])
                if bool_series is not None:
                    df[col] = bool_series
                    self.actions_.append(f"Converted '{col}': {old_dtype} -> bool")
                    continue

            # Numeric downcast
            if self.downcast_numerics and pd.api.types.is_numeric_dtype(df[col]):
                if pd.api.types.is_integer_dtype(df[col]):
                    df[col] = pd.to_numeric(df[col], downcast="integer")
                else:
                    # Only downcast floats if they don't actually need to be ints
                    if df[col].dropna().apply(float.is_integer).all():
                        df[col] = pd.to_numeric(df[col], downcast="integer")
                    else:
                        df[col] = pd.to_numeric(df[col], downcast="float")
                if df[col].dtype != old_dtype:
                    self.actions_.append(
                        f"Downcasted '{col}': {old_dtype} -> {df[col].dtype}"
                    )
                continue

            # Categorical inference
            if self.infer_categoricals and is_string_like(df[col]):
                n = len(df[col])
                if n > 0:
                    uniq = df[col].nunique(dropna=True)
                    if uniq / n <= self.category_threshold and uniq > 1:
                        df[col] = df[col].astype("category")
                        self.actions_.append(
                            f"Converted '{col}': object -> category "
                            f"({uniq} unique values)"
                        )

        after_mem = df.memory_usage(deep=True).sum()
        if before_mem > 0:
            pct = (1 - after_mem / before_mem) * 100
            self.actions_.append(
                f"Memory: {before_mem/1024:.1f} KB -> {after_mem/1024:.1f} KB "
                f"({pct:+.1f}%)"
            )
        return df

    def _try_to_bool(self, series: pd.Series) -> pd.Series | None:
        non_null = series.dropna().astype(str).str.strip().str.lower()
        if len(non_null) == 0:
            return None
        unique_vals = set(non_null.unique())
        if not unique_vals.issubset(self.BOOL_TRUE | self.BOOL_FALSE):
            return None

        def cast(val):
            if pd.isna(val):
                return val
            return str(val).strip().lower() in self.BOOL_TRUE

        return series.map(cast).astype("boolean")  # nullable bool
