"""
data_loader.py
--------------
Everything related to reading the dataset and getting it into a clean,
analysis-ready state.

Two functions matter here:
    load_data()    -> reads the CSV and parses dates
    inspect_data() -> prints the standard "first look" checks

Run from the project root:
    python src/data_loader.py
"""

import os

import pandas as pd

DEFAULT_PATH = os.path.join("data", "sales_data.csv")


def load_data(path: str = DEFAULT_PATH) -> pd.DataFrame:
    """Read the sales CSV and return a cleaned DataFrame.

    Cleaning steps:
      * Order_Date is converted from text to a real datetime column, which is
        what lets us group by month later.
      * Month and Month_Name helper columns are added.
      * Exact duplicate rows are dropped.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Could not find '{path}'. Run `python src/generate_data.py` first."
        )

    df = pd.read_csv(path)

    # Without this conversion Order_Date is just a string and .dt accessors fail.
    df["Order_Date"] = pd.to_datetime(df["Order_Date"])

    # Period('M') keeps months in true chronological order, which plain month
    # names would not ("April" would sort before "January").
    df["Month"] = df["Order_Date"].dt.to_period("M")
    df["Month_Name"] = df["Order_Date"].dt.strftime("%b %Y")

    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    removed = before - len(df)
    if removed:
        print(f"Removed {removed} duplicate row(s).")

    return df


def inspect_data(df: pd.DataFrame) -> None:
    """Print the standard set of first-look checks on the dataset."""
    print("=" * 60)
    print("DATASET INSPECTION")
    print("=" * 60)

    print("\n--- First 5 rows ---")
    print(df.head())

    print("\n--- Last 5 rows ---")
    print(df.tail())

    print("\n--- Shape (rows, columns) ---")
    print(df.shape)

    print("\n--- Column names ---")
    print(list(df.columns))

    print("\n--- Data types ---")
    print(df.dtypes)

    print("\n--- Missing values per column ---")
    print(df.isnull().sum())

    print("\n--- Duplicate records ---")
    print(df.duplicated().sum())

    print("\n--- Summary statistics (numeric columns) ---")
    print(df.describe())


if __name__ == "__main__":
    data = load_data()
    inspect_data(data)
