"""Internal helpers."""

import pandas as pd


def is_string_like(series: pd.Series) -> bool:
    """
    True if a Series holds strings, regardless of whether pandas stores
    them as 'object', the nullable 'string' dtype, or pandas 3.x 'str'.
    """
    dtype = series.dtype
    if dtype == "object":
        return True
    # pandas 1.x/2.x nullable string + pandas 3.x 'str'
    name = getattr(dtype, "name", str(dtype))
    return name in {"string", "str"} or pd.api.types.is_string_dtype(series)
