"""Handle missing values with configurable strategies per column or globally."""

from __future__ import annotations
import pandas as pd
import numpy as np
from typing import Any


class NullHandler:
    """
    Handles nulls in a DataFrame.

    Strategies:
        - 'drop'    : drop rows with any null in the target columns
        - 'mean'    : fill numeric nulls with the column mean
        - 'median'  : fill numeric nulls with the column median
        - 'mode'    : fill nulls with the most frequent value
        - 'ffill'   : forward fill
        - 'bfill'   : backward fill
        - 'constant': fill with a user-supplied constant
        - 'auto'    : numeric -> median, categorical/object -> mode
    """

    VALID_STRATEGIES = {
        "drop", "mean", "median", "mode",
        "ffill", "bfill", "constant", "auto",
    }

    def __init__(
        self,
        strategy: str | dict[str, str] = "auto",
        constant: Any = None,
        drop_threshold: float | None = None,
    ):
        """
        Args:
            strategy: A single strategy applied to all columns, OR a
                dict mapping column name -> strategy for per-column control.
            constant: Value to use when strategy == 'constant'.
            drop_threshold: If set (0.0-1.0), drop columns whose null
                fraction exceeds this threshold before imputing.
        """
        if isinstance(strategy, str) and strategy not in self.VALID_STRATEGIES:
            raise ValueError(
                f"Invalid strategy '{strategy}'. "
                f"Choose from {sorted(self.VALID_STRATEGIES)}."
            )
        self.strategy = strategy
        self.constant = constant
        self.drop_threshold = drop_threshold
        self.actions_: list[str] = []

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        self.actions_ = []

        # 1. Drop high-null columns if requested
        if self.drop_threshold is not None:
            null_frac = df.isna().mean()
            to_drop = null_frac[null_frac > self.drop_threshold].index.tolist()
            if to_drop:
                df = df.drop(columns=to_drop)
                self.actions_.append(
                    f"Dropped {len(to_drop)} columns above null threshold "
                    f"{self.drop_threshold}: {to_drop}"
                )

        # 2. Apply strategy per column
        if isinstance(self.strategy, dict):
            for col, strat in self.strategy.items():
                if col in df.columns:
                    df = self._apply_strategy(df, col, strat)
        else:
            for col in df.columns:
                df = self._apply_strategy(df, col, self.strategy)

        return df

    def _apply_strategy(
        self, df: pd.DataFrame, col: str, strategy: str
    ) -> pd.DataFrame:
        n_nulls = df[col].isna().sum()
        if n_nulls == 0:
            return df

        is_numeric = pd.api.types.is_numeric_dtype(df[col])

        # Resolve 'auto'
        effective = strategy
        if strategy == "auto":
            effective = "median" if is_numeric else "mode"

        if effective == "drop":
            before = len(df)
            df = df.dropna(subset=[col])
            self.actions_.append(
                f"Dropped {before - len(df)} rows with nulls in '{col}'"
            )
        elif effective == "mean":
            if not is_numeric:
                return df  # silently skip; mean only makes sense for numerics
            value = df[col].mean()
            df[col] = df[col].fillna(value)
            self.actions_.append(f"Filled {n_nulls} nulls in '{col}' with mean={value:.4g}")
        elif effective == "median":
            if not is_numeric:
                return df
            value = df[col].median()
            df[col] = df[col].fillna(value)
            self.actions_.append(f"Filled {n_nulls} nulls in '{col}' with median={value:.4g}")
        elif effective == "mode":
            mode_series = df[col].mode(dropna=True)
            if len(mode_series) == 0:
                return df
            value = mode_series.iloc[0]
            df[col] = df[col].fillna(value)
            self.actions_.append(f"Filled {n_nulls} nulls in '{col}' with mode={value!r}")
        elif effective == "ffill":
            df[col] = df[col].ffill()
            self.actions_.append(f"Forward-filled nulls in '{col}'")
        elif effective == "bfill":
            df[col] = df[col].bfill()
            self.actions_.append(f"Back-filled nulls in '{col}'")
        elif effective == "constant":
            df[col] = df[col].fillna(self.constant)
            self.actions_.append(
                f"Filled {n_nulls} nulls in '{col}' with constant={self.constant!r}"
            )

        return df
