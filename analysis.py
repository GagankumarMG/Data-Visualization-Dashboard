"""
analysis.py
-----------
All the Pandas aggregation logic lives here, separated from the plotting code.
Each function takes the cleaned DataFrame and returns a small result that can be
printed, saved, or handed straight to a chart.

Run from the project root:
    python src/analysis.py
"""

import pandas as pd

try:
    from .data_loader import load_data
except ImportError:  # running the file directly rather than as a package
    from data_loader import load_data


def overall_summary(df: pd.DataFrame) -> dict:
    """Headline numbers for the whole dataset."""
    total_sales = df["Sales"].sum()
    total_profit = df["Profit"].sum()

    return {
        "Total Orders": len(df),
        "Total Sales": total_sales,
        "Total Profit": total_profit,
        "Total Quantity Sold": int(df["Quantity"].sum()),
        "Average Sales per Order": df["Sales"].mean(),
        "Average Profit per Order": df["Profit"].mean(),
        # Profit margin is the single most useful derived number here: it tells
        # you how much of every rupee of revenue is actually kept.
        "Overall Profit Margin %": (total_profit / total_sales) * 100,
        "Loss-Making Orders": int((df["Profit"] < 0).sum()),
    }


def sales_by_category(df: pd.DataFrame) -> pd.DataFrame:
    """Sales, profit, quantity and margin for each product category."""
    result = df.groupby("Category").agg(
        Total_Sales=("Sales", "sum"),
        Total_Profit=("Profit", "sum"),
        Total_Quantity=("Quantity", "sum"),
        Order_Count=("Order_ID", "count"),
    )
    result["Profit_Margin_%"] = (
        result["Total_Profit"] / result["Total_Sales"] * 100
    ).round(2)
    return result.sort_values("Total_Sales", ascending=False)


def sales_by_region(df: pd.DataFrame) -> pd.DataFrame:
    """Same breakdown, but by region instead of category."""
    result = df.groupby("Region").agg(
        Total_Sales=("Sales", "sum"),
        Total_Profit=("Profit", "sum"),
        Total_Quantity=("Quantity", "sum"),
        Order_Count=("Order_ID", "count"),
    )
    result["Profit_Margin_%"] = (
        result["Total_Profit"] / result["Total_Sales"] * 100
    ).round(2)
    return result.sort_values("Total_Sales", ascending=False)


def monthly_sales(df: pd.DataFrame) -> pd.DataFrame:
    """Sales and profit per calendar month, in chronological order."""
    result = (
        df.groupby("Month")
        .agg(
            Total_Sales=("Sales", "sum"),
            Total_Profit=("Profit", "sum"),
            Order_Count=("Order_ID", "count"),
        )
        .sort_index()
    )
    # Convert the Period index to readable labels for the x-axis.
    result.index = result.index.strftime("%b %Y")
    return result


def top_products(df: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    """The n products with the highest total sales."""
    result = df.groupby("Product").agg(
        Total_Sales=("Sales", "sum"),
        Total_Profit=("Profit", "sum"),
        Units_Sold=("Quantity", "sum"),
    )
    return result.sort_values("Total_Sales", ascending=False).head(n)


def customer_type_breakdown(df: pd.DataFrame) -> pd.DataFrame:
    """How new customers compare with returning customers."""
    result = df.groupby("Customer_Type").agg(
        Total_Sales=("Sales", "sum"),
        Average_Sales=("Sales", "mean"),
        Order_Count=("Order_ID", "count"),
    )
    return result.round(2)


def category_region_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Pivot table of sales with categories as rows and regions as columns.

    This is the table behind the heatmap in visualizations.py.
    """
    return df.pivot_table(
        index="Category",
        columns="Region",
        values="Sales",
        aggfunc="sum",
        fill_value=0,
    )


def run_full_analysis(df: pd.DataFrame) -> None:
    """Print every analysis result in a readable order."""
    print("=" * 60)
    print("EXPLORATORY DATA ANALYSIS")
    print("=" * 60)

    print("\n--- Overall summary ---")
    for key, value in overall_summary(df).items():
        if isinstance(value, float):
            print(f"{key:<28}: {value:>14,.2f}")
        else:
            print(f"{key:<28}: {value:>14,}")

    print("\n--- Sales by category ---")
    print(sales_by_category(df).round(2))

    print("\n--- Sales by region ---")
    print(sales_by_region(df).round(2))

    print("\n--- Monthly sales ---")
    print(monthly_sales(df).round(2))

    print("\n--- Top 5 products by sales ---")
    print(top_products(df).round(2))

    print("\n--- New vs returning customers ---")
    print(customer_type_breakdown(df))

    print("\n--- Sales: category x region ---")
    print(category_region_matrix(df).round(2))

    print("\n--- Orders per category (value_counts) ---")
    print(df["Category"].value_counts())


if __name__ == "__main__":
    run_full_analysis(load_data())
