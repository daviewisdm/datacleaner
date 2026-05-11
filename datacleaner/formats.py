"""Correct common wrong-format issues: dates, numbers-as-strings, currencies, etc."""

from __future__ import annotations
import re
import pandas as pd
import numpy as np

from ._utils import is_string_like


# Pre-compiled patterns
_CURRENCY_RE = re.compile(r"[\$£€¥₹]|ksh|kes|usd|eur|gbp", re.IGNORECASE)
_NUMBER_CHARS_RE = re.compile(r"[,\s]")
_PHONE_KEEP_RE = re.compile(r"[^\d+]")
_EMAIL_RE = re.compile(r"^[\w\.\-]+@[\w\.\-]+\.\w+$")


class FormatCorrector:
    """
    Detects and fixes common formatting problems in object columns:
        - Dates stored as strings -> datetime
        - Numbers stored as strings with separators ("1,234.50") -> float
        - Currency strings ("Ksh 1,200", "$5.99") -> float
        - Phone numbers normalized to digits with optional leading '+'
        - Email addresses lowercased & trimmed (validation kept loose)
    """

    def __init__(
        self,
        parse_dates: bool = True,
        parse_numbers: bool = True,
        parse_currency: bool = True,
        normalize_phones: bool = True,
        clean_emails: bool = True,
        date_threshold: float = 0.8,
        number_threshold: float = 0.8,
        default_country_code: str | None = None,
    ):
        """
        Args:
            *_threshold: Minimum fraction of non-null values in a column
                that must successfully parse before the conversion is applied.
                This prevents accidentally munging a column that just happens
                to have a few date-like strings.
            default_country_code: If set (e.g. '254' for Kenya, '1' for US),
                phone numbers starting with '0' will be converted to
                international format by replacing the leading 0 with
                '+<code>'. Numbers already starting with '+' are unchanged.
        """
        self.parse_dates = parse_dates
        self.parse_numbers = parse_numbers
        self.parse_currency = parse_currency
        self.normalize_phones = normalize_phones
        self.clean_emails = clean_emails
        self.date_threshold = date_threshold
        self.number_threshold = number_threshold
        self.default_country_code = default_country_code
        self.actions_: list[str] = []

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        self.actions_ = []

        for col in df.columns:
            if not is_string_like(df[col]):
                continue

            # Try in priority order. Stop at first successful conversion.
            if self.clean_emails and self._looks_like_email(df[col]):
                df[col] = self._clean_email_column(df[col])
                self.actions_.append(f"Cleaned email format in '{col}'")
                continue

            if self.parse_currency and self._looks_like_currency(df[col]):
                converted = self._parse_currency_column(df[col])
                if converted is not None:
                    df[col] = converted
                    self.actions_.append(f"Parsed currency strings in '{col}' -> float")
                    continue

            if self.parse_numbers:
                converted = self._try_parse_numbers(df[col])
                if converted is not None:
                    df[col] = converted
                    self.actions_.append(f"Parsed numeric strings in '{col}' -> float")
                    continue

            if self.parse_dates:
                converted = self._try_parse_dates(df[col])
                if converted is not None:
                    df[col] = converted
                    self.actions_.append(f"Parsed date strings in '{col}' -> datetime")
                    continue

            if self.normalize_phones and self._looks_like_phone(col, df[col]):
                df[col] = self._normalize_phone_column(df[col], self.default_country_code)
                self.actions_.append(f"Normalized phone numbers in '{col}'")

        return df

    # ---------- detection ----------
    @staticmethod
    def _looks_like_email(series: pd.Series) -> bool:
        sample = series.dropna().astype(str).head(50)
        if len(sample) == 0:
            return False
        hits = sample.str.contains("@", regex=False).sum()
        return hits / len(sample) >= 0.8

    @staticmethod
    def _looks_like_currency(series: pd.Series) -> bool:
        sample = series.dropna().astype(str).head(50)
        if len(sample) == 0:
            return False
        hits = sample.str.contains(_CURRENCY_RE).sum()
        return hits / len(sample) >= 0.5

    @staticmethod
    def _looks_like_phone(col_name: str, series: pd.Series) -> bool:
        # Be conservative: only treat as phone if the column name hints at it.
        # Otherwise we'd clobber product codes, IDs, etc.
        name = str(col_name).lower()
        if not any(tok in name for tok in ("phone", "mobile", "tel", "msisdn", "cell")):
            return False
        sample = series.dropna().astype(str).head(50)
        if len(sample) == 0:
            return False
        # Most values should have at least 7 digits
        digit_counts = sample.apply(lambda s: sum(c.isdigit() for c in s))
        return (digit_counts >= 7).mean() >= 0.8

    # ---------- conversion helpers ----------
    def _try_parse_dates(self, series: pd.Series) -> pd.Series | None:
        non_null = series.dropna()
        if len(non_null) == 0:
            return None
        parsed = pd.to_datetime(non_null, errors="coerce", format="mixed")
        success_rate = parsed.notna().mean()
        if success_rate >= self.date_threshold:
            full = pd.to_datetime(series, errors="coerce", format="mixed")
            return full
        return None

    def _try_parse_numbers(self, series: pd.Series) -> pd.Series | None:
        non_null = series.dropna().astype(str)
        if len(non_null) == 0:
            return None
        cleaned = non_null.str.replace(_NUMBER_CHARS_RE, "", regex=True)
        parsed = pd.to_numeric(cleaned, errors="coerce")
        success_rate = parsed.notna().mean()
        if success_rate >= self.number_threshold:
            full_cleaned = series.astype(str).str.replace(
                _NUMBER_CHARS_RE, "", regex=True
            )
            return pd.to_numeric(full_cleaned, errors="coerce")
        return None

    def _parse_currency_column(self, series: pd.Series) -> pd.Series | None:
        non_null = series.dropna().astype(str)
        if len(non_null) == 0:
            return None
        stripped = (
            non_null.str.replace(_CURRENCY_RE, "", regex=True)
                    .str.replace(_NUMBER_CHARS_RE, "", regex=True)
        )
        parsed = pd.to_numeric(stripped, errors="coerce")
        if parsed.notna().mean() >= 0.7:
            full = (
                series.astype(str)
                      .str.replace(_CURRENCY_RE, "", regex=True)
                      .str.replace(_NUMBER_CHARS_RE, "", regex=True)
            )
            return pd.to_numeric(full, errors="coerce")
        return None

    @staticmethod
    def _normalize_phone_column(
        series: pd.Series, default_country_code: str | None = None
    ) -> pd.Series:
        def norm(val):
            if pd.isna(val):
                return val
            s = str(val).strip()
            has_plus = s.startswith("+")
            digits = _PHONE_KEEP_RE.sub("", s).lstrip("+")
            if has_plus:
                return "+" + digits
            # If a country code is supplied and the number starts with 0,
            # convert to international format (drop the 0, prepend +<code>).
            if default_country_code and digits.startswith("0"):
                return "+" + default_country_code + digits[1:]
            return digits
        return series.map(norm)

    @staticmethod
    def _clean_email_column(series: pd.Series) -> pd.Series:
        def norm(val):
            if pd.isna(val):
                return val
            s = str(val).strip().lower()
            return s if _EMAIL_RE.match(s) else np.nan
        return series.map(norm)
