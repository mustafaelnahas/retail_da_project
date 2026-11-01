# Retail Sales Analysis — Graduation Project

## Business scenario
A household-goods retailer operates three branches and wants to understand sales, profitability, discounting, product/category performance, and differences between branches. The data is synthetic and intended only for training.

## Files
- `retail_sales_raw.csv`: intentionally messy raw transactions.
- `app.py`: interactive Streamlit dashboard.
- `analysis_starter.py`: guided examples using NumPy, Pandas, Matplotlib, Seaborn, and Plotly.
- `trainer_guide.pdf`: two-hour lesson plan, KPI definitions, guided code, prompts, and recommendation cues.
- `requirements.txt`: required Python packages.

## Run the dashboard
1. Install Python 3.10+.
2. Open a terminal in this folder.
3. Run `pip install -r requirements.txt`
4. Run `streamlit run app.py`

Keep `retail_sales_raw.csv` in the same folder as `app.py`.

## Important
The data contains deliberate data-quality issues. The cleaning approach in the dashboard is a teaching baseline. In a real business, suspicious prices, discounts, and sales should be checked against receipts or source systems before records are changed.
