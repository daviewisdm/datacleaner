"""Prove the one-liner works."""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from datacleaner import clean

df = pd.DataFrame({
    "  Customer Name  ": ["Alice Mwangi", "BOB OTIENO", "  alice mwangi  ",
                          "Carol Wanjiku", "David Kimani", None,
                          "Eve Achieng", "Frank Mutua", "Grace Njeri", "Henry Kipchoge"],
    "Email Address": ["ALICE@example.com  ", "bob@example.com", "alice@example.com",
                      "carol@example.COM", "not-an-email", "eve@example.com",
                      "eve@example.com", "frank@example.com", None, "henry@example.com"],
    "Phone": ["+254-712-345-678", "0712 345 679", "+254712345680",
              "(0712) 345-681", "0712345682", "+254 712 345 683",
              "0712-345-684", None, "0712345686", "+254712345687"],
    "Price (KES)": ["Ksh 1,200", "KES 2,500.50", "1,200", "Ksh 999", "Ksh 4,300.00",
                    None, "Ksh 1,200", "2,800", "Ksh 5,500", "3,100.75"],
    "Signup Date": ["2024-01-15", "15/02/2024", "2024-03-10", "April 5, 2024",
                    "2024-05-20", "2024-06-01", "2024-07-10", None,
                    "2024-09-12", "2024-10-01"],
    "Age": [28, 35, None, 42, 31, 29, None, 38, 45, 33],
    "City": ["Nairobi", "nairobi", "Nairobi", "Mombasa", "Kisumu",
             "Nairobi", "Mombasa", "Nakuru", "Nairobi", "Kisumu"],
    "Active": ["yes", "Yes", "YES", "no", "No", "yes", "yes", "no", "yes", "no"],
    "Useless Column": [None] * 10,
})

# THE ONE LINE
cleaned = clean(df)

print("CLEANED:")
print(cleaned)
print("\nDtypes:")
print(cleaned.dtypes)

# With report
print("\n--- Same call but with verbose=True ---\n")
clean(df, verbose=True)
