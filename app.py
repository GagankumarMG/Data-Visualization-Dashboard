"""
app.py
------
Optional interactive version of the dashboard, built with Streamlit.

The charts are the exact same functions used by the notebook and the static
PNG export, so nothing is duplicated: the sidebar filters the DataFrame, and the
filtered DataFrame is passed straight into the existing plotting functions.

Run from the project root:
    streamlit run dashboard/app.py
"""

import os
import sys

import matplotlib.pyplot as plt
import streamlit as st

# Make the src/ package importable when Streamlit runs this file directly.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

from analysis import (  # noqa: E402
    monthly_sales,
    overall_summary,
    sales_by_category,
    sales_by_region,
    top_products,
)
from data_loader import load_data  # noqa: E402
from visualizations import (  # noqa: E402
    plot_category_region_heatmap,
    plot_category_sales,
    plot_monthly_sales,
    plot_regional_sales,
    plot_sales_profit_scatter,
    plot_top_products,
)

st.set_page_config(page_title="Sales Dashboard", page_icon="📊", layout="wide")


@st.cache_data
def get_data():
    """Load the CSV once and keep it cached across reruns."""
    return load_data(os.path.join(PROJECT_ROOT, "data", "sales_data.csv"))


df = get_data()

# ---------------------------------------------------------------------------
# Sidebar filters
# ---------------------------------------------------------------------------
st.sidebar.title("Filters")

categories = st.sidebar.multiselect(
    "Category",
    options=sorted(df["Category"].unique()),
    default=sorted(df["Category"].unique()),
)
regions = st.sidebar.multiselect(
    "Region",
    options=sorted(df["Region"].unique()),
    default=sorted(df["Region"].unique()),
)
customer_types = st.sidebar.multiselect(
    "Customer Type",
    options=sorted(df["Customer_Type"].unique()),
    default=sorted(df["Customer_Type"].unique()),
)

filtered = df[
    df["Category"].isin(categories)
    & df["Region"].isin(regions)
    & df["Customer_Type"].isin(customer_types)
]

st.title("📊 Sales Performance Dashboard")
st.caption("Built with Pandas, Matplotlib and Seaborn")

if filtered.empty:
    st.warning("No records match the current filters. Widen your selection.")
    st.stop()

# ---------------------------------------------------------------------------
# KPI row
# ---------------------------------------------------------------------------
summary = overall_summary(filtered)
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Total Sales", f"{summary['Total Sales']:,.0f}")
c2.metric("Total Profit", f"{summary['Total Profit']:,.0f}")
c3.metric("Profit Margin", f"{summary['Overall Profit Margin %']:.1f}%")
c4.metric("Units Sold", f"{summary['Total Quantity Sold']:,}")
c5.metric("Orders", f"{summary['Total Orders']:,}")

st.divider()

# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------
row1_left, row1_right = st.columns(2)
with row1_left:
    st.pyplot(plot_category_sales(filtered, save=False))
with row1_right:
    st.pyplot(plot_monthly_sales(filtered, save=False))

row2_left, row2_right = st.columns(2)
with row2_left:
    st.pyplot(plot_regional_sales(filtered, save=False))
with row2_right:
    st.pyplot(plot_sales_profit_scatter(filtered, save=False))

row3_left, row3_right = st.columns(2)
with row3_left:
    st.pyplot(plot_category_region_heatmap(filtered, save=False))
with row3_right:
    st.pyplot(plot_top_products(filtered, save=False))

# Streamlit keeps figures in memory otherwise, which leaks across reruns.
plt.close("all")

st.divider()

# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs(
    ["By Category", "By Region", "Monthly", "Raw Data"]
)
with tab1:
    st.dataframe(sales_by_category(filtered).round(2), use_container_width=True)
with tab2:
    st.dataframe(sales_by_region(filtered).round(2), use_container_width=True)
with tab3:
    st.dataframe(monthly_sales(filtered).round(2), use_container_width=True)
with tab4:
    st.dataframe(
        filtered.drop(columns=["Month"]), use_container_width=True, hide_index=True
    )

st.sidebar.divider()
st.sidebar.subheader("Top products")
st.sidebar.dataframe(top_products(filtered, n=5).round(0), use_container_width=True)
