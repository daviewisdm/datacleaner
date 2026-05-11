# datacleaner

Automated data cleaning for pandas DataFrames. Handles nulls, duplicates, text standardization, format correction (dates, currencies, numbers stored as strings, phone numbers, emails), and dtype optimization — all in one call, or step-by-step.

## Install

```bash
pip install -e .
```

## Quick start (one line)

```python
import pandas as pd
from datacleaner import clean

clean_df = clean(pd.read_csv("messy.csv"))
```

That's it. The `clean()` function auto-detects and fixes everything — nulls, duplicates, messy text, wrong formats, bad dtypes.

Want to see what changed?

```python
clean_df = clean(df, verbose=True)
```

## Need more control?

```python
from datacleaner import DataCleaner

cleaner = DataCleaner(df)
clean_df = cleaner.clean()
print(cleaner.report())
```

## Configuring individual steps

```python
cleaner = DataCleaner(
    df,
    standardize_config={"text_case": "lower"},
    null_config={
        "strategy": {"age": "median", "city": "mode"},  # per-column
        "drop_threshold": 0.5,                          # drop cols >50% null
    },
    duplicate_config={"subset": ["email"], "normalize_strings": True},
    dtype_config={"category_threshold": 0.3},
)
clean_df = cleaner.clean()
```

## Using a single step

```python
from datacleaner import NullHandler
filled = NullHandler(strategy="median").fit_transform(df)
```

## Pipeline order

1. **Standardize** — clean column names, trim whitespace, optionally lowercase
2. **Formats** — parse dates, currencies, numeric strings; normalize phones/emails
3. **Nulls** — impute or drop, per-column or globally
4. **Duplicates** — exact or case/whitespace-insensitive
5. **Dtypes** — downcast numerics, infer categoricals, infer booleans

Order matters: standardizing text BEFORE deduplicating means `"John Doe"` and `"  john doe "` collapse correctly. Parsing formats BEFORE imputing means `mean()` on a "price" column works because it's already numeric.

## API

- `DataCleaner(df, **configs).clean() -> pd.DataFrame`
- `DataCleaner(df).report() -> str`
- Each step exposed individually: `NullHandler`, `DuplicateHandler`, `Standardizer`, `FormatCorrector`, `DTypeConverter`
