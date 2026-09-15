"""
generate_data.py
----------------
Creates the sample dataset used by this project: data/sales_data.csv

The random seed is fixed (SEED = 42) so that every run produces exactly the
same 90 records. This makes the whole project reproducible: anyone who clones
the repository and runs this file will get the same numbers and the same charts.

Run from the project root:
    python src/generate_data.py
"""

import os

import numpy as np
import pandas as pd

SEED = 42
N_RECORDS = 90
OUTPUT_PATH = os.path.join("data", "sales_data.csv")

# Each product belongs to exactly one category, and has a realistic unit price
# range. Keeping this mapping in one place avoids impossible rows such as a
# "Sofa" appearing under "Clothing".
PRODUCT_CATALOG = {
    "Electronics": {
        "Laptop": (45000, 85000),
        "Smartphone": (12000, 45000),
        "Headphones": (1500, 9000),
        "Monitor": (8000, 22000),
    },
    "Furniture": {
        "Office Chair": (4000, 14000),
        "Study Table": (5000, 18000),
        "Bookshelf": (3000, 11000),
        "Sofa": (15000, 40000),
    },
    "Clothing": {
        "T-Shirt": (400, 1500),
        "Jeans": (900, 3500),
        "Jacket": (1800, 6500),
        "Formal Shirt": (800, 2800),
    },
    "Accessories": {
        "Backpack": (900, 3500),
        "Wrist Watch": (1200, 9000),
        "Sunglasses": (700, 4000),
        "Wallet": (400, 2000),
    },
}

REGIONS = ["North", "South", "East", "West"]
CUSTOMER_TYPES = ["New", "Returning"]

# Profit margin varies by category, which is what makes the profit analysis
# interesting: the highest-selling category is not automatically the most
# profitable one.
CATEGORY_MARGIN = {
    "Electronics": (0.06, 0.14),
    "Furniture": (0.10, 0.20),
    "Clothing": (0.18, 0.32),
    "Accessories": (0.20, 0.38),
}

# A gentle upward trend across the year plus some seasonality, so the monthly
# line chart has a story to tell instead of being pure noise.
MONTH_WEIGHTS = np.array(
    [0.9, 0.85, 1.0, 1.05, 1.0, 0.95, 1.1, 1.05, 1.15, 1.25, 1.4, 1.3]
)


def generate_dataset(n_records: int = N_RECORDS, seed: int = SEED) -> pd.DataFrame:
    """Build a reproducible sales DataFrame with `n_records` rows."""
    rng = np.random.default_rng(seed)

    categories = list(PRODUCT_CATALOG.keys())
    rows = []

    # Spread the orders across 2024 using the month weights above.
    month_probs = MONTH_WEIGHTS / MONTH_WEIGHTS.sum()
    months = rng.choice(np.arange(1, 13), size=n_records, p=month_probs)

    for i, month in enumerate(months, start=1):
        # Day 1-28 keeps every generated date valid, including February.
        day = int(rng.integers(1, 29))
        order_date = pd.Timestamp(year=2024, month=int(month), day=day)

        category = str(rng.choice(categories, p=[0.30, 0.22, 0.26, 0.22]))
        product = str(rng.choice(list(PRODUCT_CATALOG[category].keys())))

        low, high = PRODUCT_CATALOG[category][product]
        unit_price = float(rng.uniform(low, high))
        quantity = int(rng.integers(1, 6))
        sales = round(unit_price * quantity, 2)

        margin_low, margin_high = CATEGORY_MARGIN[category]
        margin = float(rng.uniform(margin_low, margin_high))

        # Roughly one order in twelve is sold at a loss (discounting, returns,
        # clearance). Without these the scatter plot would be a perfect line.
        if rng.random() < 0.08:
            margin = -float(rng.uniform(0.02, 0.09))

        profit = round(sales * margin, 2)

        rows.append(
            {
                "Order_ID": f"ORD{1000 + i}",
                "Order_Date": order_date.strftime("%Y-%m-%d"),
                "Product": product,
                "Category": category,
                "Region": str(rng.choice(REGIONS, p=[0.28, 0.24, 0.23, 0.25])),
                "Sales": sales,
                "Quantity": quantity,
                "Profit": profit,
                "Customer_Type": str(rng.choice(CUSTOMER_TYPES, p=[0.45, 0.55])),
            }
        )

    df = pd.DataFrame(rows)
    return df.sort_values("Order_Date").reset_index(drop=True)


def main() -> None:
    df = generate_dataset()
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Created {OUTPUT_PATH} with {len(df)} records.")
    print(df.head())


if __name__ == "__main__":
    main()
