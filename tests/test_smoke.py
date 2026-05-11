"""Smoke test: build a dataset with EVERY problem, run the cleaner, inspect."""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from datacleaner import DataCleaner

# A deliberately ugly dataset
df = pd.DataFrame({
    "  Customer Name  ": [
        "Alice Mwangi", "BOB OTIENO", "  alice mwangi  ",   # dup w/ different casing
        "Carol Wanjiku", "David Kimani", None,
        "Eve Achieng", "Frank Mutua", "Grace Njeri", "Henry Kipchoge",
    ],
    "Email Address": [
        "ALICE@example.com  ", "bob@example.com", "alice@example.com",
        "carol@example.COM", "not-an-email", "eve@example.com",
        "eve@example.com", "frank@example.com", None, "henry@example.com",
    ],
    "Phone": [
        "+254-712-345-678", "0712 345 679", "+254712345680",
        "(0712) 345-681", "0712345682", "+254 712 345 683",
        "0712-345-684", None, "0712345686", "+254712345687",
    ],
    "Price (KES)": [
        "Ksh 1,200", "KES 2,500.50", "1,200",
        "Ksh 999", "Ksh 4,300.00", None,
        "Ksh 1,200", "2,800", "Ksh 5,500", "3,100.75",
    ],
    "Signup Date": [
        "2024-01-15", "15/02/2024", "2024-03-10",
        "April 5, 2024", "2024-05-20", "2024-06-01",
        "2024-07-10", None, "2024-09-12", "2024-10-01",
    ],
    "Age": [
        28, 35, None, 42, 31, 29, None, 38, 45, 33,
    ],
    "City": [
        "Nairobi", "nairobi", "Nairobi",
        "Mombasa", "Kisumu", "Nairobi",
        "Mombasa", "Nakuru", "Nairobi", "Kisumu",
    ],
    "Active": [
        "yes", "Yes", "YES", "no", "No", "yes",
        "yes", "no", "yes", "no",
    ],
    "Useless Column": [None] * 10,   # 100% null, should be dropped
})

print("ORIGINAL:")
print(df)
print("\nDtypes:")
print(df.dtypes)
print("\n" + "=" * 60)

cleaner = DataCleaner(
    df,
    null_config={"drop_threshold": 0.9},  # drop the 100% null col
    duplicate_config={"subset": ["customer_name", "email_address"]},
    dtype_config={"category_threshold": 0.6},  # cities are repetitive but few rows
)
clean = cleaner.clean()

print("\nCLEANED:")
print(clean)
print("\nDtypes:")
print(clean.dtypes)
print()
print(cleaner.report())

# Assertions to catch regressions
assert "useless_column" not in clean.columns, "100% null col should be dropped"
assert "customer_name" in clean.columns, "column should be snake_cased"
assert pd.api.types.is_datetime64_any_dtype(clean["signup_date"]), "dates not parsed"
assert pd.api.types.is_numeric_dtype(clean["price_kes"]), "currency not parsed"
assert clean["active"].dtype == "boolean", f"booleans not inferred (got {clean['active'].dtype})"
assert clean["city"].dtype.name == "category", f"category not inferred (got {clean['city'].dtype})"
assert clean["age"].isna().sum() == 0, "age nulls not imputed"
assert len(clean) < len(df), "duplicates not removed"

print("\n✓ All assertions passed.")
