import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

st.set_page_config(page_title="Retail Sales Analysis", layout="wide")
st.title("Retail Sales Analysis")
st.caption("Graduation project • Synthetic retail transactions • EGP")

@st.cache_data
def load_data():
    return pd.read_csv("retail_sales_raw.csv")

raw = load_data()
df = raw.copy()
df["Order_Date"] = pd.to_datetime(df["Order_Date"], errors="coerce")
rows_before = len(df)
df = df.drop_duplicates()

# Validate numeric fields
for col in ["Quantity", "Unit_Price_EGP", "Discount_Rate", "Sales_EGP", "Cost_EGP", "Profit_EGP"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")
df.loc[(df["Discount_Rate"] < 0) | (df["Discount_Rate"] > 0.8), "Discount_Rate"] = np.nan
df.loc[df["Unit_Price_EGP"] <= 0, "Unit_Price_EGP"] = np.nan
df.loc[df["Quantity"] <= 0, "Quantity"] = np.nan
df["Payment_Method"] = df["Payment_Method"].fillna("Unknown")
df["Customer_Type"] = df["Customer_Type"].fillna("Unknown")
df["Discount_Rate"] = df["Discount_Rate"].fillna(df["Discount_Rate"].median())
df["Unit_Price_EGP"] = df["Unit_Price_EGP"].fillna(df.groupby("Category")["Unit_Price_EGP"].transform("median"))
df["Quantity"] = df["Quantity"].fillna(df["Quantity"].median())
# Flag suspicious sales values; do not silently treat negative sales as valid
q99 = df["Sales_EGP"].quantile(.99)
df["Sales_EGP"] = df["Sales_EGP"].clip(lower=0, upper=q99)
# Recalculate derived values after cleaning; keep source cost as the available cost estimate
df["Sales_EGP"] = (df["Unit_Price_EGP"] * df["Quantity"] * (1 - df["Discount_Rate"])).round(2)
df["Profit_EGP"] = (df["Sales_EGP"] - df["Cost_EGP"]).round(2)
df["Order_Month"] = df["Order_Date"].dt.to_period("M").astype(str)
df["Profit_Margin"] = np.where(df["Sales_EGP"] > 0, df["Profit_EGP"] / df["Sales_EGP"], np.nan)

st.sidebar.header("Filters")
branches = ["All"] + sorted(df["Branch"].dropna().unique().tolist())
branch = st.sidebar.selectbox("Branch", branches)
categories = ["All"] + sorted(df["Category"].dropna().unique().tolist())
category = st.sidebar.selectbox("Category", categories)
payments = ["All"] + sorted(df["Payment_Method"].dropna().unique().tolist())
payment = st.sidebar.selectbox("Payment method", payments)
customer_types = ["All"] + sorted(df["Customer_Type"].dropna().unique().tolist())
customer_type = st.sidebar.selectbox("Customer type", customer_types)

f = df.copy()
if branch != "All": f = f[f["Branch"] == branch]
if category != "All": f = f[f["Category"] == category]
if payment != "All": f = f[f["Payment_Method"] == payment]
if customer_type != "All": f = f[f["Customer_Type"] == customer_type]

total_sales = f["Sales_EGP"].sum()
total_profit = f["Profit_EGP"].sum()
orders = f["Transaction_ID"].nunique()
units = f["Quantity"].sum()
margin = (total_profit / total_sales * 100) if total_sales else 0
avg_basket = f["Sales_EGP"].mean() if len(f) else 0

c1,c2,c3,c4,c5 = st.columns(5)
c1.metric("Sales (EGP)", f"{total_sales:,.0f}")
c2.metric("Profit (EGP)", f"{total_profit:,.0f}")
c3.metric("Transactions", f"{orders:,}")
c4.metric("Profit margin", f"{margin:.1f}%")
c5.metric("Avg. transaction", f"{avg_basket:,.0f}")

st.subheader("Sales overview")
left, right = st.columns(2)
with left:
    monthly = f.groupby("Order_Month", as_index=False).agg(Sales_EGP=("Sales_EGP","sum"))
    fig = px.line(monthly, x="Order_Month", y="Sales_EGP", markers=True, title="Monthly sales trend")
    st.plotly_chart(fig, use_container_width=True)
with right:
    cat_sales = f.groupby("Category", as_index=False).agg(Sales_EGP=("Sales_EGP","sum"))
    fig = px.bar(cat_sales, x="Category", y="Sales_EGP", color="Category", title="Sales by category")
    st.plotly_chart(fig, use_container_width=True)

st.subheader("Profitability & branches")
left, right = st.columns(2)
with left:
    product_perf = f.groupby("Product", as_index=False).agg(
        Sales_EGP=("Sales_EGP","sum"), Profit_EGP=("Profit_EGP","sum"), Units=("Quantity","sum")
    ).sort_values("Profit_EGP", ascending=False)
    fig = px.bar(product_perf.head(10), x="Profit_EGP", y="Product", orientation="h",
                 color="Profit_EGP", title="Top 10 products by profit")
    st.plotly_chart(fig, use_container_width=True)
with right:
    branch_perf = f.groupby("Branch", as_index=False).agg(
        Sales_EGP=("Sales_EGP","sum"), Profit_EGP=("Profit_EGP","sum"),
        Transactions=("Transaction_ID","nunique")
    )
    fig = px.scatter(branch_perf, x="Sales_EGP", y="Profit_EGP", size="Transactions",
                     color="Branch", hover_name="Branch", title="Branch sales vs profit")
    st.plotly_chart(fig, use_container_width=True)

st.subheader("Discount and margin")
fig = px.scatter(f, x="Discount_Rate", y="Profit_Margin", color="Category",
                 hover_data=["Product","Sales_EGP","Profit_EGP"],
                 title="Discount rate vs profit margin")
st.plotly_chart(fig, use_container_width=True)

with st.expander("Data quality checks"):
    st.write(f"Rows before removing exact duplicates: {rows_before:,}")
    st.write(f"Rows after removing exact duplicates: {len(df):,}")
    st.write("Remaining missing values:")
    st.dataframe(df.isna().sum().rename("Missing values").to_frame())
    st.caption("Synthetic data note: suspicious sales values are recalculated from cleaned price, quantity, and discount. In real business data, investigate anomalies against source receipts before overwriting.")

with st.expander("Explore filtered data"):
    st.dataframe(f, use_container_width=True)
    st.download_button("Download filtered data", f.to_csv(index=False).encode("utf-8"),
                       file_name="filtered_retail_sales.csv", mime="text/csv")

st.subheader("Suggested business actions")
st.markdown("""
- Protect profit, not revenue alone: prioritize products with strong absolute profit and sustainable margins.
- Review discounts where the discount rate is high but profit margin is low.
- Compare branches using sales, profit, and transaction count together.
- Investigate sudden monthly sales changes before attributing them to marketing or seasonality.
- Validate unusual prices and sales against receipts; do not silently delete or cap business anomalies.
""")
