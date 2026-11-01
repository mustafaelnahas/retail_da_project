import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# 1) Import and inspect
raw = pd.read_csv("retail_sales_raw.csv")
print(raw.shape)
print(raw.head())
print(raw.info())
print(raw.isna().sum())
print("Exact duplicates:", raw.duplicated().sum())

# 2) NumPy: robust summary and percentile
sales_values = pd.to_numeric(raw["Sales_EGP"], errors="coerce").dropna().to_numpy()
print("Median sales:", np.median(sales_values))
print("95th percentile:", np.percentile(sales_values, 95))

# 3) Clean and validate
df = raw.copy()
df["Order_Date"] = pd.to_datetime(df["Order_Date"], errors="coerce")
df = df.drop_duplicates()
for col in ["Quantity", "Unit_Price_EGP", "Discount_Rate", "Sales_EGP", "Cost_EGP", "Profit_EGP"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")
df.loc[(df["Discount_Rate"] < 0) | (df["Discount_Rate"] > .8), "Discount_Rate"] = np.nan
df.loc[df["Unit_Price_EGP"] <= 0, "Unit_Price_EGP"] = np.nan
df.loc[df["Quantity"] <= 0, "Quantity"] = np.nan
df["Payment_Method"] = df["Payment_Method"].fillna("Unknown")
df["Customer_Type"] = df["Customer_Type"].fillna("Unknown")
df["Discount_Rate"] = df["Discount_Rate"].fillna(df["Discount_Rate"].median())
df["Unit_Price_EGP"] = df["Unit_Price_EGP"].fillna(
    df.groupby("Category")["Unit_Price_EGP"].transform("median"))
df["Quantity"] = df["Quantity"].fillna(df["Quantity"].median())
df["Sales_EGP"] = (df["Unit_Price_EGP"] * df["Quantity"] * (1-df["Discount_Rate"])).round(2)
df["Profit_EGP"] = (df["Sales_EGP"] - df["Cost_EGP"]).round(2)
df["Order_Month"] = df["Order_Date"].dt.to_period("M").astype(str)
df["Profit_Margin"] = np.where(df["Sales_EGP"] > 0, df["Profit_EGP"]/df["Sales_EGP"], np.nan)

# 4) Analysis
print("Sales by category:\\n", df.groupby("Category").agg(
    Sales=("Sales_EGP","sum"), Profit=("Profit_EGP","sum"), Units=("Quantity","sum")
).sort_values("Profit", ascending=False).round(2))
print("Branch performance:\\n", df.groupby("Branch").agg(
    Sales=("Sales_EGP","sum"), Profit=("Profit_EGP","sum"),
    Transactions=("Transaction_ID","nunique")
).round(2))
print("Correlation matrix:\\n", df[["Quantity","Unit_Price_EGP","Discount_Rate","Sales_EGP","Profit_EGP"]].corr().round(2))

# 5) Matplotlib
df.groupby("Category")["Sales_EGP"].sum().sort_values().plot(kind="barh", title="Sales by category")
plt.xlabel("Sales (EGP)")
plt.tight_layout()
plt.show()

# 6) Seaborn
sns.boxplot(data=df, x="Category", y="Profit_Margin")
plt.title("Profit margin distribution by category")
plt.xticks(rotation=20)
plt.tight_layout()
plt.show()

# 7) Plotly
monthly = df.groupby("Order_Month", as_index=False).agg(Sales_EGP=("Sales_EGP","sum"))
fig = px.line(monthly, x="Order_Month", y="Sales_EGP", markers=True, title="Monthly sales trend")
fig.show()
