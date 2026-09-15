"""
visualizations.py
-----------------
All plotting code. Every function saves a PNG into outputs/ and returns the
Matplotlib figure, so the same functions can be reused by the notebook and by
the Streamlit app without duplicating chart code.

Run from the project root to regenerate every image:
    python src/visualizations.py
"""

import os

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# "Agg" is a non-interactive backend. It lets the script save PNG files on a
# machine with no display attached, which is how this runs in CI or a server.
matplotlib.use("Agg")

try:
    from .analysis import (
        category_region_matrix,
        monthly_sales,
        sales_by_category,
        sales_by_region,
        top_products,
    )
    from .data_loader import load_data
except ImportError:
    from analysis import (
        category_region_matrix,
        monthly_sales,
        sales_by_category,
        sales_by_region,
        top_products,
    )
    from data_loader import load_data

OUTPUT_DIR = "outputs"

# One consistent look for every chart in the project. Setting this once at
# import time is cleaner than repeating styling arguments in each function.
PALETTE = ["#2E5EAA", "#5BA85A", "#E08A3C", "#B5485D", "#6C5B9E"]
sns.set_theme(style="whitegrid", palette=PALETTE)
plt.rcParams.update(
    {
        "figure.dpi": 110,
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
        "font.size": 11,
        "axes.titlesize": 14,
        "axes.titleweight": "bold",
        "axes.labelsize": 11,
        "axes.edgecolor": "#CCCCCC",
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    }
)


def _ensure_output_dir() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def _save(fig: plt.Figure, filename: str) -> str:
    """Save a figure into outputs/ and report the path."""
    _ensure_output_dir()
    path = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(path)
    print(f"Saved {path}")
    return path


def _thousands(value: float, _pos=None) -> str:
    """Format large axis numbers as 1.2K / 3.4M so labels stay readable."""
    if abs(value) >= 1_000_000:
        return f"{value / 1_000_000:.1f}M"
    if abs(value) >= 1_000:
        return f"{value / 1_000:.0f}K"
    return f"{value:.0f}"


# ---------------------------------------------------------------------------
# Visualization 1 - Sales by Category (bar chart)
# ---------------------------------------------------------------------------
def plot_category_sales(df: pd.DataFrame, ax: plt.Axes = None, save: bool = True):
    """Bar chart of total sales per product category, highest first."""
    data = sales_by_category(df)

    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(8, 5))
    else:
        fig = ax.figure

    bars = ax.bar(
        data.index,
        data["Total_Sales"],
        color=PALETTE[: len(data)],
        edgecolor="white",
        linewidth=1.2,
    )

    # Data labels on top of each bar so the reader does not have to trace values
    # back to the axis.
    for bar, value in zip(bars, data["Total_Sales"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            _thousands(value),
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    ax.set_title("Total Sales by Product Category")
    ax.set_xlabel("Category")
    ax.set_ylabel("Total Sales")
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(_thousands))
    ax.margins(y=0.15)

    if standalone and save:
        _save(fig, "category_sales.png")
    return fig


# ---------------------------------------------------------------------------
# Visualization 2 - Monthly Sales Trend (line chart)
# ---------------------------------------------------------------------------
def plot_monthly_sales(df: pd.DataFrame, ax: plt.Axes = None, save: bool = True):
    """Line chart of sales month by month across the year."""
    data = monthly_sales(df)

    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(9, 5))
    else:
        fig = ax.figure

    ax.plot(
        data.index,
        data["Total_Sales"],
        marker="o",
        markersize=6,
        linewidth=2.2,
        color=PALETTE[0],
        label="Sales",
    )
    # A light fill under the line makes the peaks and dips easier to read.
    ax.fill_between(data.index, data["Total_Sales"], alpha=0.12, color=PALETTE[0])

    # A dashed mean line gives every month a reference point to compare against.
    mean_sales = data["Total_Sales"].mean()
    ax.axhline(
        mean_sales,
        color="#888888",
        linestyle="--",
        linewidth=1.2,
        label=f"Monthly average ({_thousands(mean_sales)})",
    )

    ax.set_title("Monthly Sales Trend")
    # Inside the dashboard grid the rotated date ticks already reach the panel
    # below, so the redundant "Month" label is dropped there.
    ax.set_xlabel("Month" if standalone else "")
    ax.set_ylabel("Total Sales")
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(_thousands))
    ax.tick_params(axis="x", rotation=45)
    ax.legend(frameon=False, fontsize=9)

    if standalone and save:
        _save(fig, "monthly_sales.png")
    return fig


# ---------------------------------------------------------------------------
# Visualization 3 - Regional Sales (horizontal bar chart)
# ---------------------------------------------------------------------------
def plot_regional_sales(df: pd.DataFrame, ax: plt.Axes = None, save: bool = True):
    """Horizontal bars comparing total sales across the four regions."""
    data = sales_by_region(df).sort_values("Total_Sales")

    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(8, 5))
    else:
        fig = ax.figure

    # The best-performing region is highlighted; the rest are muted, so the
    # answer to "which region won?" is visible at a glance.
    colors = ["#B8C4D9"] * (len(data) - 1) + [PALETTE[0]]
    bars = ax.barh(data.index, data["Total_Sales"], color=colors, edgecolor="white")

    for bar, value in zip(bars, data["Total_Sales"]):
        ax.text(
            bar.get_width() * 1.01,
            bar.get_y() + bar.get_height() / 2,
            _thousands(value),
            va="center",
            fontsize=10,
            fontweight="bold",
        )

    ax.set_title("Total Sales by Region")
    ax.set_xlabel("Total Sales")
    ax.set_ylabel("Region")
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(_thousands))
    ax.margins(x=0.15)

    if standalone and save:
        _save(fig, "regional_sales.png")
    return fig


# ---------------------------------------------------------------------------
# Visualization 4 - Sales vs Profit (scatter plot)
# ---------------------------------------------------------------------------
def plot_sales_profit_scatter(
    df: pd.DataFrame, ax: plt.Axes = None, save: bool = True
):
    """Scatter plot of every order: sales on x, profit on y, coloured by category."""
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(8.5, 5.5))
    else:
        fig = ax.figure

    sns.scatterplot(
        data=df,
        x="Sales",
        y="Profit",
        hue="Category",
        size="Quantity",
        sizes=(40, 220),
        alpha=0.75,
        edgecolor="white",
        linewidth=0.6,
        ax=ax,
    )

    # The break-even line: anything below it is an order sold at a loss.
    ax.axhline(0, color="#B5485D", linestyle="--", linewidth=1.2, zorder=0)

    ax.set_title("Sales vs Profit by Order")
    ax.set_xlabel("Sales (per order)")
    ax.set_ylabel("Profit (per order)")
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(_thousands))
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(_thousands))
    ax.legend(fontsize=8, frameon=False, loc="upper left", ncol=2)

    if standalone and save:
        _save(fig, "sales_profit_scatter.png")
    return fig


# ---------------------------------------------------------------------------
# Visualization 5 - Category x Region heatmap
# ---------------------------------------------------------------------------
def plot_category_region_heatmap(
    df: pd.DataFrame, ax: plt.Axes = None, save: bool = True
):
    """Heatmap showing which category sells best in which region."""
    matrix = category_region_matrix(df)

    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(8, 5))
    else:
        fig = ax.figure

    sns.heatmap(
        matrix,
        annot=True,
        fmt=".0f",
        cmap="Blues",
        linewidths=0.6,
        linecolor="white",
        cbar_kws={"label": "Sales"},
        ax=ax,
    )

    ax.set_title("Sales Heatmap: Category vs Region")
    ax.set_xlabel("Region")
    ax.set_ylabel("Category")

    if standalone and save:
        _save(fig, "category_region_heatmap.png")
    return fig


# ---------------------------------------------------------------------------
# Visualization 6 - Top products
# ---------------------------------------------------------------------------
def plot_top_products(df: pd.DataFrame, ax: plt.Axes = None, save: bool = True):
    """Horizontal bar chart of the five best-selling products."""
    data = top_products(df, n=5).sort_values("Total_Sales")

    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(8, 5))
    else:
        fig = ax.figure

    ax.barh(data.index, data["Total_Sales"], color=PALETTE[4], edgecolor="white")
    for y, value in enumerate(data["Total_Sales"]):
        ax.text(
            value * 1.01,
            y,
            _thousands(value),
            va="center",
            fontsize=10,
            fontweight="bold",
        )

    ax.set_title("Top 5 Products by Sales")
    ax.set_xlabel("Total Sales")
    ax.set_ylabel("Product")
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(_thousands))
    ax.margins(x=0.15)

    if standalone and save:
        _save(fig, "top_products.png")
    return fig


# ---------------------------------------------------------------------------
# Combined dashboard
# ---------------------------------------------------------------------------
def build_dashboard(df: pd.DataFrame, save: bool = True) -> plt.Figure:
    """Arrange the six charts into a single dashboard image.

    GridSpec is used instead of plain subplots so the KPI strip at the top can
    span the full width while the charts below sit in a regular grid.
    """
    fig = plt.figure(figsize=(18, 14))
    grid = fig.add_gridspec(
        nrows=4,
        ncols=2,
        height_ratios=[0.35, 1, 1, 1],
        hspace=0.55,
        wspace=0.22,
    )

    fig.suptitle(
        "Sales Performance Dashboard",
        fontsize=24,
        fontweight="bold",
        y=0.97,
    )

    # --- KPI strip -----------------------------------------------------------
    kpi_ax = fig.add_subplot(grid[0, :])
    kpi_ax.axis("off")

    total_sales = df["Sales"].sum()
    total_profit = df["Profit"].sum()
    kpis = [
        ("Total Sales", _thousands(total_sales)),
        ("Total Profit", _thousands(total_profit)),
        ("Profit Margin", f"{total_profit / total_sales * 100:.1f}%"),
        ("Units Sold", f"{int(df['Quantity'].sum())}"),
        ("Total Orders", f"{len(df)}"),
    ]

    for i, (label, value) in enumerate(kpis):
        x = (i + 0.5) / len(kpis)
        kpi_ax.text(
            x, 0.62, value, ha="center", va="center",
            fontsize=22, fontweight="bold", color=PALETTE[0],
        )
        kpi_ax.text(
            x, 0.18, label.upper(), ha="center", va="center",
            fontsize=10, color="#666666",
        )

    # --- Charts --------------------------------------------------------------
    plot_category_sales(df, ax=fig.add_subplot(grid[1, 0]), save=False)
    plot_monthly_sales(df, ax=fig.add_subplot(grid[1, 1]), save=False)
    plot_regional_sales(df, ax=fig.add_subplot(grid[2, 0]), save=False)
    plot_sales_profit_scatter(df, ax=fig.add_subplot(grid[2, 1]), save=False)
    plot_category_region_heatmap(df, ax=fig.add_subplot(grid[3, 0]), save=False)
    plot_top_products(df, ax=fig.add_subplot(grid[3, 1]), save=False)

    if save:
        _save(fig, "dashboard.png")
    return fig


def generate_all(df: pd.DataFrame = None) -> None:
    """Regenerate every PNG in outputs/."""
    if df is None:
        df = load_data()

    print("Generating visualizations...")
    plot_category_sales(df)
    plot_monthly_sales(df)
    plot_regional_sales(df)
    plot_sales_profit_scatter(df)
    plot_category_region_heatmap(df)
    plot_top_products(df)
    build_dashboard(df)
    plt.close("all")
    print("All charts written to outputs/")


if __name__ == "__main__":
    generate_all()
