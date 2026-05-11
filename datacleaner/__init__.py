"""
datacleaner — Automated data cleaning for pandas DataFrames.

Quick start:
    >>> import pandas as pd
    >>> from datacleaner import DataCleaner
    >>> df = pd.read_csv("messy.csv")
    >>> cleaner = DataCleaner(df)
    >>> clean_df = cleaner.clean()           # run the full pipeline
    >>> print(cleaner.report())              # see what changed
"""

from .cleaner import DataCleaner
from .nulls import NullHandler
from .duplicates import DuplicateHandler
from .standardize import Standardizer
from .formats import FormatCorrector
from .dtypes import DTypeConverter

import pandas as pd


def clean(df: pd.DataFrame, verbose: bool = False, **overrides) -> pd.DataFrame:
    """
    One-line data cleaning. Auto-detects and fixes everything.

    Runs the full pipeline with sensible defaults:
        - Standardizes column names (snake_case), trims whitespace,
          and applies smart per-column casing (Title Case for names/locations,
          lowercase for emails/URLs, untouched for free text and IDs)
        - Parses dates, currencies, numeric strings; normalizes phones/emails
        - Imputes nulls (median for numerics, mode for text) and drops columns >90% null
        - Removes duplicates (case- and whitespace-insensitive)
        - Optimizes dtypes (downcasts numerics, infers booleans & categoricals)

    Args:
        df: The DataFrame to clean. Not modified.
        verbose: If True, prints a report of every change made.
        **overrides: Optional per-step overrides, e.g.
            standardize_config={...}, null_config={...}, etc.

    Returns:
        A new cleaned DataFrame.

    Example:
        >>> import pandas as pd
        >>> from datacleaner import clean
        >>> clean_df = clean(pd.read_csv("messy.csv"))
    """
    defaults = {
        "standardize_config": {"smart_case": True},
        "null_config": {"strategy": "auto", "drop_threshold": 0.9},
        "duplicate_config": {"normalize_strings": True},
    }
    defaults.update(overrides)

    cleaner = DataCleaner(df, **defaults)
    result = cleaner.clean()
    if verbose:
        print(cleaner.report())
    return result


__version__ = "0.2.0"
__all__ = [
    "clean",
    "DataCleaner",
    "NullHandler",
    "DuplicateHandler",
    "Standardizer",
    "FormatCorrector",
    "DTypeConverter",
]
