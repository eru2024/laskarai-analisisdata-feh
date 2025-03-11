# Import Packages/Libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from babel.numbers import format_currency
sns.set(style='dark')
import numpy as np
import gdown
import geopandas as gpd
from googletrans import Translator
from datetime import datetime
translator = Translator()

import streamlit as st
import pandas as pd
import gdown
from datetime import datetime

# --- Data Loading Function ---
@st.cache_data
def load_data():
    file_id = '1f-Jz07Cn3md5j91Zrti84nH2UTjhAywO'
    url = f'https://drive.google.com/uc?id={file_id}'
    output = 'orders_datasets.csv'
    gdown.download(url, output, quiet=False)
    df = pd.read_csv(output)
    # Convert purchase timestamp to datetime for filtering
    df['order_purchase_timestamp'] = pd.to_datetime(df['order_purchase_timestamp'])
    return df

# Load dataset
df_orders = load_data()

# --- Sidebar: Date Filter ---
st.sidebar.header("Pilih rentang waktu")
min_date = df_orders['order_purchase_timestamp'].min().date()
max_date = df_orders['order_purchase_timestamp'].max().date()

start_date = st.sidebar.date_input("Start Date", value=min_date, min_value=min_date, max_value=max_date)
end_date = st.sidebar.date_input("End Date", value=max_date, min_value=min_date, max_value=max_date)

# Filter dataset based on the selected date range
df_filtered = df_orders[
    (df_orders['order_purchase_timestamp'] >= pd.to_datetime(start_date)) &
    (df_orders['order_purchase_timestamp'] <= pd.to_datetime(end_date))
].copy()

# --- Main Header and Text ---
st.title("E-Commerce Dashboard")
st.write("Created by Fatih El Haq | Cohort LaskarAi | ID a002ybf161")
st.subheader("Orders Overview")

# --- Visualization 1: Total Orders & Yearly Growth Orders (Left-Aligned) ---
total_orders = len(df_filtered)

# Add a 'year' column for yearly grouping
df_filtered['year'] = df_filtered['order_purchase_timestamp'].dt.year
orders_by_year = df_filtered.groupby('year').size().reset_index(name='orders').sort_values('year')

# Calculate the yearly growth between the latest year and the previous year
if len(orders_by_year) >= 2:
    latest_year = orders_by_year['year'].max()
    previous_year = latest_year - 1
    orders_latest = orders_by_year.loc[orders_by_year['year'] == latest_year, 'orders'].values[0]
    if previous_year in orders_by_year['year'].values:
        orders_previous = orders_by_year.loc[orders_by_year['year'] == previous_year, 'orders'].values[0]
        growth_percent = ((orders_latest - orders_previous) / orders_previous * 100) if orders_previous != 0 else 0
    else:
        growth_percent = 0
else:
    latest_year = previous_year = None
    growth_percent = 0

# Use a simple two-column layout (default is left-aligned)
col1, col2 = st.columns(2)
with col1:
    st.metric("Total Orders", total_orders)
with col2:
    if latest_year and previous_year:
        st.metric(f"Yearly Growth Orders [{latest_year}-{previous_year}]", f"{growth_percent:.2f}%")
    else:
        st.metric("Yearly Growth Orders", f"{growth_percent:.2f}%")

# --- Visualization 2: Order Status Metrics (Left-Aligned) ---
def get_status_metric(df, status):
    count = df[df['order_status'].str.lower() == status.lower()].shape[0]
    percent = (count / total_orders * 100) if total_orders > 0 else 0
    return count, percent

# Compute metrics for each status
invoiced_count, invoiced_percent = get_status_metric(df_filtered, "invoiced")
unavailable_count, unavailable_percent = get_status_metric(df_filtered, "unavailable")
canceled_count, canceled_percent = get_status_metric(df_filtered, "canceled")
shipped_count, shipped_percent = get_status_metric(df_filtered, "shipped")
delivered_count, delivered_percent = get_status_metric(df_filtered, "delivered")

# Render each metric with left-aligned text.
def render_metric(label, count, percent, highlight=False):
    if highlight:
        html = f"""
        <div style='background-color: red; padding: 10px; border-radius: 5px; text-align: left; color: white;'>
            <strong>{label}</strong><br>
            <span style='font-size:24px;'>{count}</span><br>
            <span style='font-size:12px;'>{percent:.2f}%</span>
        </div>
        """
    else:
        html = f"""
        <div style='padding: 10px; border-radius: 5px; text-align: left;'>
            <strong>{label}</strong><br>
            <span style='font-size:24px;'>{count}</span><br>
            <span style='font-size:12px;'>{percent:.2f}%</span>
        </div>
        """
    st.markdown(html, unsafe_allow_html=True)

# Display metrics in five columns
col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    render_metric("Invoiced", invoiced_count, invoiced_percent, highlight=False)
with col2:
    render_metric("Unavailable", unavailable_count, unavailable_percent, highlight=True)
with col3:
    render_metric("Canceled", canceled_count, canceled_percent, highlight=True)
with col4:
    render_metric("Shipped", shipped_count, shipped_percent, highlight=False)
with col5:
    render_metric("Delivered", delivered_count, delivered_percent, highlight=False)

# --- Visualization 3: Monthly Orders Growth (Line Chart) ---

# One line gap
st.markdown("<br>", unsafe_allow_html=True)

# Create a month column from the filtered data
df_filtered['order_purchase_month'] = df_filtered['order_purchase_timestamp'].dt.to_period('M')
monthly_order_counts = df_filtered.groupby('order_purchase_month')['order_id'].count()

fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(monthly_order_counts.index.astype(str), monthly_order_counts.values, marker='o')

# Modify x-axis labels to display month (first line) and year (second line)
x_labels = [
    f"{pd.Period(m, freq='M').strftime('%b')}\n{pd.Period(m, freq='M').strftime('%Y')}"
    for m in monthly_order_counts.index
]
ax.set_xticks(range(len(x_labels)))
ax.set_xticklabels(x_labels, rotation=0)

# Find highest and lowest points
max_index = monthly_order_counts.idxmax()
min_index = monthly_order_counts.idxmin()
max_value = monthly_order_counts[max_index]
min_value = monthly_order_counts[min_index]
max_x = monthly_order_counts.index.get_loc(max_index)
min_x = monthly_order_counts.index.get_loc(min_index)

ax.annotate(
    f'{max_value}', (max_x, max_value),
    xytext=(max_x - 1.25, max_value - 100),
    textcoords='data', ha='center', fontsize=12, fontweight='bold',
    bbox=dict(boxstyle='round,pad=0.5', fc='lightblue', ec='black', lw=1),
    arrowprops=dict(arrowstyle='wedge,tail_width=0.7', fc='lightblue', ec='black')
)

ax.annotate(
    f'{min_value}', (min_x, min_value),
    xytext=(min_x, min_value + 500),
    textcoords='data', ha='center', fontsize=12, fontweight='bold',
    bbox=dict(boxstyle='round,pad=0.5', fc='lightblue', ec='black', lw=1),
    arrowprops=dict(arrowstyle='wedge,tail_width=0.7', fc='lightblue', ec='black')
)

sns.despine(top=True, right=True, ax=ax)
ax.set_xlabel('Order Purchase Month')
ax.set_ylabel('Number of Orders')
ax.set_title('Monthly Order Count')
fig.tight_layout()
st.pyplot(fig)

# --- Visualization 4: Purchase Patterns (Composite Chart) ---
st.subheader("Purchase Patterns")

# Use a copy of filtered data for purchase patterns
df_purchase_pattern = df_filtered.copy()

# 1) Phase of the Month
df_purchase_pattern['order_purchase_day'] = df_purchase_pattern['order_purchase_timestamp'].dt.day
df_purchase_pattern['month_phase'] = pd.cut(
    df_purchase_pattern['order_purchase_day'],
    bins=[0, 10, 20, 31],
    labels=['Beginning', 'Mid', 'End']
)
order_month_phase_counts = df_purchase_pattern['month_phase'].value_counts().reindex(['Beginning', 'Mid', 'End'])
order_month_phase_percentages = (order_month_phase_counts / len(df_purchase_pattern)) * 100

# 2) Time of Day (Day Part)
df_purchase_pattern['order_purchase_day_time'] = pd.cut(
    df_purchase_pattern['order_purchase_timestamp'].dt.hour,
    bins=[0, 6, 12, 18, 24],
    labels=['Dawn', 'Morning', 'Afternoon', 'Night'],
    right=False
)
order_purchase_day_counts = df_purchase_pattern['order_purchase_day_time'].value_counts().reindex(['Dawn', 'Morning', 'Afternoon', 'Night'])
order_purchase_day_percentages = (order_purchase_day_counts / len(df_purchase_pattern)) * 100

# 3) Day of the Week (Mon–Sun)
df_purchase_pattern['order_day_of_week'] = df_purchase_pattern['order_purchase_timestamp'].dt.dayofweek
day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
df_purchase_pattern['order_day_name'] = df_purchase_pattern['order_day_of_week'].map(lambda x: day_names[x])
order_purchase_week_counts = df_purchase_pattern['order_day_name'].value_counts().reindex(day_names)
order_purchase_week_percentages = (order_purchase_week_counts / len(df_purchase_pattern)) * 100

# Create a composite figure with 2 rows x 2 columns layout
fig2 = plt.figure(figsize=(12, 6.5))
gs = fig2.add_gridspec(2, 2)

# Subplot 1: Total Orders by Phase of the Month (horizontal)
ax1_v4 = fig2.add_subplot(gs[0, 0])
bars1 = sns.countplot(
    y='month_phase',
    data=df_purchase_pattern,
    order=['Beginning', 'Mid', 'End'],
    color="#00008B",
    ax=ax1_v4
)
for bar, count, percentage in zip(bars1.patches, order_month_phase_counts, order_month_phase_percentages):
    width = bar.get_width()
    height = bar.get_height()
    x, y = bar.get_xy()
    ax1_v4.text(x + width / 2, y + height / 2, f'{count: ,}', ha='center', va='center', color='white')
    ax1_v4.text(width + 200, y + height / 2, f'{percentage:.1f}%', ha='left', va='center', color='black')
sns.despine(ax=ax1_v4, left=True, bottom=True)
ax1_v4.set_xticks([])
ax1_v4.set_xlabel('Number of Orders')
ax1_v4.set_ylabel('')
ax1_v4.set_title('Total Orders by Phase of the Month')

# Subplot 2: Total Orders by Time of the Day (horizontal)
ax2_v4 = fig2.add_subplot(gs[0, 1])
bars2 = sns.countplot(
    y='order_purchase_day_time',
    data=df_purchase_pattern,
    order=['Dawn', 'Morning', 'Afternoon', 'Night'],
    color="#00008B",
    ax=ax2_v4
)
for bar, count, percentage in zip(bars2.patches, order_purchase_day_counts, order_purchase_day_percentages):
    width = bar.get_width()
    height = bar.get_height()
    x, y = bar.get_xy()
    ax2_v4.text(x + width / 2, y + height / 2, f'{count: ,}', ha='center', va='center', color='white')
    ax2_v4.text(width + 200, y + height / 2, f'{percentage:.1f}%', ha='left', va='center', color='black')
y_labels = ['Dawn\n00.00','Morning\n06.00','Afternoon\n12.00','Night\n18.00']
ax2_v4.set_yticks(range(len(y_labels)))
ax2_v4.set_yticklabels(y_labels, rotation=0)
sns.despine(ax=ax2_v4, left=True, bottom=True)
ax2_v4.set_xticks([])
ax2_v4.set_xlabel('Number of Orders')
ax2_v4.set_ylabel('')
ax2_v4.set_title('Total Orders by Time of the Day')

# Subplot 3: Total Orders by Day of the Week (spanning entire second row)
ax3_v4 = fig2.add_subplot(gs[1, :])
bars3 = sns.countplot(
    x='order_day_name',
    data=df_purchase_pattern,
    order=day_names,
    color="#00008B",
    ax=ax3_v4
)
for bar, count, percentage in zip(bars3.patches, order_purchase_week_counts, order_purchase_week_percentages):
    height = bar.get_height()
    x_center = bar.get_x() + bar.get_width() / 2
    ax3_v4.text(x_center, height, f'{percentage:.1f}%', ha='center', va='bottom')
    bars3.annotate(f'{count: ,}', (x_center, height - 200),
                   ha='center', va='top', fontsize=10, color='white')
x_labels = ['Mon','Tue','Wed','Thu','Fri','Sat','Sun']
ax3_v4.set_xticks(range(len(x_labels)))
ax3_v4.set_xticklabels(x_labels, rotation=0)
sns.despine(ax=ax3_v4, left=True)
ax3_v4.set_yticks([])
ax3_v4.set_xlabel('Day of the Week')
ax3_v4.set_ylabel('Number of Orders')
ax3_v4.set_title('Total Orders by Day of the Week')

plt.tight_layout()
st.pyplot(fig2)