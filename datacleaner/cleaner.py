"""High-level orchestrator that runs the full cleaning pipeline."""

from __future__ import annotations
import pandas as pd
from typing import Any

from .standardize import Standardizer
from .formats import FormatCorrector
from .dtypes import DTypeConverter
from .nulls import NullHandler
from .duplicates import DuplicateHandler


class DataCleaner:
    """
    One-call orchestrator for cleaning a DataFrame.

    Pipeline order (matters!):
        1. Standardize   — fix column names & text BEFORE matching on values
        2. Formats       — turn '1,200', '12/03/2024', 'Ksh 500' into real types
        3. Nulls         — now that types are right, imputation makes sense
        4. Duplicates    — after normalization, duplicates collapse correctly
        5. Dtypes        — final memory & type optimization

    Example:
        >>> cleaner = DataCleaner(df)
        >>> clean = cleaner.clean()
        >>> print(cleaner.report())

        Override any step's config:
        >>> cleaner = DataCleaner(
        ...     df,
        ...     null_config={"strategy": {"age": "median", "city": "mode"}},
        ...     duplicate_config={"subset": ["email"]},
        ... )
    """

    def __init__(
        self,
        df: pd.DataFrame,
        *,
        standardize_config: dict[str, Any] | None = None,
        format_config: dict[str, Any] | None = None,
        null_config: dict[str, Any] | None = None,
        duplicate_config: dict[str, Any] | None = None,
        dtype_config: dict[str, Any] | None = None,
        steps: list[str] | None = None,
    ):
        """
        Args:
            df: Input DataFrame (not modified).
            *_config: kwargs forwarded to each step's constructor.
            steps: Subset/order of steps to run. Default runs all in
                the recommended order:
                ['standardize', 'formats', 'nulls', 'duplicates', 'dtypes']
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"Expected pandas DataFrame, got {type(df).__name__}")

        self.original_df = df
        self.cleaned_df: pd.DataFrame | None = None

        self.steps = steps or [
            "standardize", "formats", "nulls", "duplicates", "dtypes",
        ]

        self.standardizer = Standardizer(**(standardize_config or {}))
        self.format_corrector = FormatCorrector(**(format_config or {}))
        self.null_handler = NullHandler(**(null_config or {}))
        self.duplicate_handler = DuplicateHandler(**(duplicate_config or {}))
        self.dtype_converter = DTypeConverter(**(dtype_config or {}))

        self._step_map = {
            "standardize": self.standardizer,
            "formats": self.format_corrector,
            "nulls": self.null_handler,
            "duplicates": self.duplicate_handler,
            "dtypes": self.dtype_converter,
        }

    def clean(self) -> pd.DataFrame:
        df = self.original_df.copy()
        for step in self.steps:
            if step not in self._step_map:
                raise ValueError(
                    f"Unknown step '{step}'. Valid: {list(self._step_map)}"
                )
            df = self._step_map[step].fit_transform(df)
        self.cleaned_df = df
        return df

    def report(self) -> str:
        """Human-readable summary of every change made."""
        if self.cleaned_df is None:
            return "No cleaning has been run yet. Call .clean() first."

        before_rows, before_cols = self.original_df.shape
        after_rows, after_cols = self.cleaned_df.shape
        before_nulls = int(self.original_df.isna().sum().sum())
        after_nulls = int(self.cleaned_df.isna().sum().sum())

        lines = [
            "=" * 60,
            "DATA CLEANING REPORT",
            "=" * 60,
            f"Shape:    {before_rows} x {before_cols}  ->  {after_rows} x {after_cols}",
            f"Nulls:    {before_nulls}  ->  {after_nulls}",
            "",
        ]
        for step in self.steps:
            handler = self._step_map[step]
            actions = getattr(handler, "actions_", [])
            lines.append(f"[{step.upper()}]")
            if actions:
                for a in actions:
                    lines.append(f"  • {a}")
            else:
                lines.append("  • (no changes)")
            lines.append("")
        lines.append("=" * 60)
        return "\n".join(lines)
