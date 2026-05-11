"""Standardize column names and text values."""

from __future__ import annotations
import re
import pandas as pd

from ._utils import is_string_like


# Keywords in (cleaned) column names that hint at the column's purpose.
# Used by smart_case to pick the right casing automatically.
_NAME_HINTS = {
    "name", "customer", "client", "user", "employee", "person",
    "first", "last", "full", "contact", "owner",
}
_LOCATION_HINTS = {
    "city", "country", "region", "state", "province", "county",
    "town", "district", "location", "address", "area", "neighborhood",
}
_EMAIL_HINTS = {"email", "mail", "e_mail"}
_ID_HINTS = {"id", "code", "sku", "ref", "reference", "uuid", "guid"}
_URL_HINTS = {"url", "link", "website", "domain"}


class Standardizer:
    """
    Standardizes:
        - Column names (snake_case, no special chars)
        - String values (strip whitespace, collapse internal spaces)
        - Casing — either globally (text_case) or smartly per column (smart_case)
    """

    def __init__(
        self,
        clean_column_names: bool = True,
        strip_strings: bool = True,
        collapse_whitespace: bool = True,
        text_case: str | None = None,        # 'lower', 'upper', 'title', or None
        smart_case: bool = False,            # per-column heuristic casing
    ):
        if text_case not in {None, "lower", "upper", "title"}:
            raise ValueError("text_case must be None, 'lower', 'upper', or 'title'")
        if smart_case and text_case is not None:
            raise ValueError(
                "Use either smart_case or text_case, not both. "
                "smart_case applies per-column casing; text_case applies one casing to all."
            )
        self.clean_column_names = clean_column_names
        self.strip_strings = strip_strings
        self.collapse_whitespace = collapse_whitespace
        self.text_case = text_case
        self.smart_case = smart_case
        self.actions_: list[str] = []

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        self.actions_ = []

        if self.clean_column_names:
            new_cols = {c: self._clean_col_name(c) for c in df.columns}
            renamed = {k: v for k, v in new_cols.items() if k != v}
            if renamed:
                df = df.rename(columns=new_cols)
                self.actions_.append(f"Renamed {len(renamed)} column(s) to snake_case")

        smart_actions: list[str] = []

        # Operate on string-like columns (object, string, str)
        for col in df.columns:
            if not is_string_like(df[col]):
                continue
            series = df[col]
            if self.strip_strings:
                series = series.str.strip()
            if self.collapse_whitespace:
                series = series.str.replace(r"\s+", " ", regex=True)

            # Casing
            if self.smart_case:
                case_choice = self._infer_case(col, series)
                if case_choice == "title":
                    series = series.str.title()
                elif case_choice == "lower":
                    series = series.str.lower()
                elif case_choice == "upper":
                    series = series.str.upper()
                if case_choice:
                    smart_actions.append(f"'{col}' -> {case_choice}")
            elif self.text_case == "lower":
                series = series.str.lower()
            elif self.text_case == "upper":
                series = series.str.upper()
            elif self.text_case == "title":
                series = series.str.title()

            df[col] = series

        if self.strip_strings or self.collapse_whitespace:
            self.actions_.append(
                f"Standardized text in object columns "
                f"(strip={self.strip_strings}, collapse={self.collapse_whitespace})"
            )
        if self.smart_case and smart_actions:
            self.actions_.append("Applied smart casing: " + ", ".join(smart_actions))
        elif self.text_case:
            self.actions_.append(f"Applied '{self.text_case}' case to all string columns")

        return df

    # ---------- helpers ----------
    @staticmethod
    def _clean_col_name(name: str) -> str:
        """Lowercase, replace non-alphanumerics with underscores, collapse repeats."""
        name = str(name).strip().lower()
        name = re.sub(r"[^\w]+", "_", name)
        name = re.sub(r"_+", "_", name)
        name = name.strip("_")
        return name or "unnamed"

    @staticmethod
    def _infer_case(col_name: str, series: pd.Series) -> str | None:
        """
        Pick a casing strategy based on column name and sample values.

        Rules (first match wins):
            1. Emails / URLs -> lowercase (case-insensitive identifiers)
            2. Plain IDs/codes -> leave as-is (often case-sensitive)
            3. Content-based: values look like emails -> lowercase
            4. Column name hints at people's names -> title case
            5. Column name hints at locations -> title case
            6. Categorical-like short values -> title case
            7. Long free text -> leave as-is
        """
        tokens = set(re.split(r"_", col_name.lower()))

        if tokens & _EMAIL_HINTS or tokens & _URL_HINTS:
            return "lower"
        if tokens & _ID_HINTS:
            return None

        sample = series.dropna().astype(str).head(20)
        if len(sample) > 0:
            email_hits = sample.str.contains("@", regex=False).sum()
            if email_hits / len(sample) >= 0.8:
                return "lower"

        if tokens & _NAME_HINTS:
            return "title"
        if tokens & _LOCATION_HINTS:
            return "title"

        if len(sample) > 0:
            avg_len = sample.str.len().mean()
            unique_ratio = series.nunique(dropna=True) / max(len(series.dropna()), 1)
            if avg_len <= 25 and unique_ratio <= 0.5:
                return "title"

        return None
