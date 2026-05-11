"""Detect and remove duplicate rows."""

from __future__ import annotations
import pandas as pd

from ._utils import is_string_like


class DuplicateHandler:
    """
    Removes duplicate rows.

    Modes:
        - exact (default): pandas .duplicated() on chosen subset
        - normalized: lowercase & strip whitespace on string columns
          BEFORE comparing, so "John Doe" == "  john doe  "
    """

    def __init__(
        self,
        subset: list[str] | None = None,
        keep: str = "first",
        normalize_strings: bool = True,
    ):
        """
        Args:
            subset: Columns to consider when detecting duplicates.
                None means use all columns.
            keep: 'first', 'last', or False (drop all duplicates).
            normalize_strings: If True, compare string columns case- and
                whitespace-insensitively. The OUTPUT keeps the original values.
        """
        self.subset = subset
        self.keep = keep
        self.normalize_strings = normalize_strings
        self.actions_: list[str] = []

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        self.actions_ = []
        before = len(df)

        if self.normalize_strings:
            # Build a comparison frame without mutating the original
            compare_df = df.copy()
            cols_to_consider = self.subset or compare_df.columns.tolist()
            for col in cols_to_consider:
                if col in compare_df.columns and is_string_like(compare_df[col]):
                    compare_df[col] = (
                        compare_df[col].astype(str).str.strip().str.lower()
                    )
            mask = compare_df.duplicated(subset=self.subset, keep=self.keep)
            df = df[~mask].copy()
        else:
            df = df.drop_duplicates(subset=self.subset, keep=self.keep).copy()

        removed = before - len(df)
        if removed:
            self.actions_.append(
                f"Removed {removed} duplicate row(s) "
                f"(subset={self.subset or 'all'}, keep={self.keep!r}, "
                f"normalized={self.normalize_strings})"
            )
        return df.reset_index(drop=True)
