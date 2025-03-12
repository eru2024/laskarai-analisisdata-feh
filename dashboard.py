# --- Import Packages and Libraries ---
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
import asyncio

# --- Data Loading Function for Orders ---
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

# --- Load Orders Dataset ---
df_orders = load_data()

# --- Determine Date Range for Filtering ---
min_date = df_orders['order_purchase_timestamp'].min().date()
max_date = df_orders['order_purchase_timestamp'].max().date()

# --- Main Header and Subheader ---
st.title("E-Commerce Dashboard")
st.write("Created by Fatih El Haq | Cohort LaskarAi | ID a002ybf161")

start_date, end_date = st.date_input(
    label='Pilih rentang waktu order dibuat',
    min_value=min_date,
    max_value=max_date,
    value=[min_date, max_date]
)

# --- Filter Orders Based on Selected Date Range ---
df_filtered = df_orders[
    (df_orders['order_purchase_timestamp'] >= pd.to_datetime(start_date)) &
    (df_orders['order_purchase_timestamp'] <= pd.to_datetime(end_date))
].copy()

st.subheader("Orders Overview")

# --- Visualization 1: Total Orders & Yearly Growth Orders (Left-Aligned) ---
total_orders = len(df_filtered)
df_filtered['year'] = df_filtered['order_purchase_timestamp'].dt.year
orders_by_year = df_filtered.groupby('year').size().reset_index(name='orders').sort_values('year')

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

invoiced_count, invoiced_percent = get_status_metric(df_filtered, "invoiced")
unavailable_count, unavailable_percent = get_status_metric(df_filtered, "unavailable")
canceled_count, canceled_percent = get_status_metric(df_filtered, "canceled")
shipped_count, shipped_percent = get_status_metric(df_filtered, "shipped")
delivered_count, delivered_percent = get_status_metric(df_filtered, "delivered")

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

col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    render_metric("Invoiced", invoiced_count, invoiced_percent)
with col2:
    render_metric("Unavailable", unavailable_count, unavailable_percent, highlight=True)
with col3:
    render_metric("Canceled", canceled_count, canceled_percent, highlight=True)
with col4:
    render_metric("Shipped", shipped_count, shipped_percent)
with col5:
    render_metric("Delivered", delivered_count, delivered_percent)

# --- Visualization 3: Monthly Orders Growth (Line Chart) ---
st.markdown("<br>", unsafe_allow_html=True)
df_filtered['order_purchase_month'] = df_filtered['order_purchase_timestamp'].dt.to_period('M')
monthly_order_counts = df_filtered.groupby('order_purchase_month')['order_id'].count()

fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(monthly_order_counts.index.astype(str), monthly_order_counts.values, marker='o')
x_labels = [
    f"{pd.Period(m, freq='M').strftime('%b')}\n{pd.Period(m, freq='M').strftime('%Y')}"
    for m in monthly_order_counts.index
]
ax.set_xticks(range(len(x_labels)))
ax.set_xticklabels(x_labels, rotation=0)
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
df_purchase_pattern = df_filtered.copy()

# Phase of the Month
df_purchase_pattern['order_purchase_day'] = df_purchase_pattern['order_purchase_timestamp'].dt.day
df_purchase_pattern['month_phase'] = pd.cut(
    df_purchase_pattern['order_purchase_day'],
    bins=[0, 10, 20, 31],
    labels=['Beginning', 'Mid', 'End']
)
order_month_phase_counts = df_purchase_pattern['month_phase'].value_counts().reindex(['Beginning', 'Mid', 'End'])
order_month_phase_percentages = (order_month_phase_counts / len(df_purchase_pattern)) * 100

# Time of Day (Day Part)
df_purchase_pattern['order_purchase_day_time'] = pd.cut(
    df_purchase_pattern['order_purchase_timestamp'].dt.hour,
    bins=[0, 6, 12, 18, 24],
    labels=['Dawn', 'Morning', 'Afternoon', 'Night'],
    right=False
)
order_purchase_day_counts = df_purchase_pattern['order_purchase_day_time'].value_counts().reindex(['Dawn', 'Morning', 'Afternoon', 'Night'])
order_purchase_day_percentages = (order_purchase_day_counts / len(df_purchase_pattern)) * 100

# Day of the Week
df_purchase_pattern['order_day_of_week'] = df_purchase_pattern['order_purchase_timestamp'].dt.dayofweek
day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
df_purchase_pattern['order_day_name'] = df_purchase_pattern['order_day_of_week'].map(lambda x: day_names[x])
order_purchase_week_counts = df_purchase_pattern['order_day_name'].value_counts().reindex(day_names)
order_purchase_week_percentages = (order_purchase_week_counts / len(df_purchase_pattern)) * 100

fig2 = plt.figure(figsize=(12, 6.5))
gs = fig2.add_gridspec(2, 2)

# Subplot 1: Total Orders by Phase of the Month
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
    ax1_v4.text(x + width/2, y + height/2, f'{count: ,}', ha='center', va='center', color='white')
    ax1_v4.text(width + 200, y + height/2, f'{percentage:.1f}%', ha='left', va='center', color='black')
sns.despine(ax=ax1_v4, left=True, bottom=True)
ax1_v4.set_xticks([])
ax1_v4.set_xlabel('Number of Orders')
ax1_v4.set_ylabel('')
ax1_v4.set_title('Total Orders by Phase of the Month')

# Subplot 2: Total Orders by Time of the Day
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
    ax2_v4.text(x + width/2, y + height/2, f'{count: ,}', ha='center', va='center', color='white')
    ax2_v4.text(width + 200, y + height/2, f'{percentage:.1f}%', ha='left', va='center', color='black')
y_labels = ['Dawn\n00.00','Morning\n06.00','Afternoon\n12.00','Night\n18.00']
ax2_v4.set_yticks(range(len(y_labels)))
ax2_v4.set_yticklabels(y_labels, rotation=0)
sns.despine(ax=ax2_v4, left=True, bottom=True)
ax2_v4.set_xticks([])
ax2_v4.set_xlabel('Number of Orders')
ax2_v4.set_ylabel('')
ax2_v4.set_title('Total Orders by Time of the Day')

# Subplot 3: Total Orders by Day of the Week
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
    x_center = bar.get_x() + bar.get_width()/2
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

# --- Visualization 5: Product Category Performance ---
st.subheader("Product Category Performance")

# Load orders_items dataset for visualization 5
@st.cache_data
def load_orders_items():
    file_id = '1EtiXssNEqUgP2QF0nUhee5AdLyQmW4v0'
    url = f'https://drive.google.com/uc?id={file_id}'
    output = 'orders_items_datasets.csv'
    gdown.download(url, output, quiet=False)
    df_items = pd.read_csv(output)
    # Create an item_id if not present
    if 'item_id' not in df_items.columns:
        df_items['item_id'] = df_items['order_id'].astype(str) + '_' + df_items['order_item_id'].astype(str)
    # Convert order_purchase_timestamp to datetime for filtering
    df_items['order_purchase_timestamp'] = pd.to_datetime(df_items['order_purchase_timestamp'])
    return df_items

df_orders_items_ref = load_orders_items()

# Filter orders_items based on selected date range
df_orders_items = df_orders_items_ref[
    (df_orders_items_ref['order_purchase_timestamp'] >= pd.to_datetime(start_date)) &
    (df_orders_items_ref['order_purchase_timestamp'] <= pd.to_datetime(end_date))
].copy()

# Merge with orders to ensure customer_id is present
if 'customer_id' not in df_orders_items.columns:
    df_orders_items = pd.merge(df_orders_items, df_orders[['order_id', 'customer_id']], on='order_id', how='left')

# Group by product category and calculate summary metrics
category_summary = df_orders_items.groupby('product_category_name_english').agg(
    total_items_ordered=('item_id', 'count'),
    total_order_value=('price', 'sum'),
    total_orders=('order_id', 'nunique'),
    total_customers=('customer_id', 'nunique')
).reset_index()

category_summary['items_per_order'] = category_summary['total_items_ordered'] / category_summary['total_orders']

# Sort and select segments for performance analysis
best_items_ordered = category_summary.sort_values('total_items_ordered', ascending=False).head(5)
worst_items_ordered = category_summary.sort_values('total_items_ordered', ascending=True).head(5)
best_order_value = category_summary.sort_values('total_order_value', ascending=False).head(5)
worst_order_value = category_summary.sort_values('total_order_value', ascending=True).head(5)
best_items_per_order = category_summary.sort_values('items_per_order', ascending=False).head(10)

# Group by order_id to aggregate unique product categories for category pairs
order_product_categories = df_orders_items.groupby('order_id')['product_category_name_english'].unique().reset_index()
multiple_categories_orders = order_product_categories[order_product_categories['product_category_name_english'].apply(len) > 1]
orders_pair_categories = pd.DataFrame({
    'order_id': multiple_categories_orders['order_id'],
    'product_category_pair': multiple_categories_orders['product_category_name_english'].apply(lambda x: ', '.join(sorted(x)))
})
category_pair_counts = orders_pair_categories.groupby('product_category_pair')['order_id'].count().reset_index()
category_pair_counts = category_pair_counts.rename(columns={'order_id': 'total_orders'})
top_10_product_category_pairs = category_pair_counts.sort_values(by='total_orders', ascending=False).head(10)

# Create composite figure for product category performance (3 rows x 2 columns)
fig3, axes = plt.subplots(nrows=3, ncols=2, figsize=(12, 15))

# Subplot 1: Best 5 Product Categories by Total Items Ordered
ax = axes[0, 0]
bars = ax.barh(best_items_ordered['product_category_name_english'],
               best_items_ordered['total_items_ordered'],
               color='#00008B')
for bar in bars:
    width = bar.get_width()
    label_y = bar.get_y() + bar.get_height()/2
    ax.text(width/2, label_y, f'{int(width): ,}', ha='center', va='center', color='white')
ax.set_xlabel('Total Items Ordered')
ax.set_title('Best 5 Product Categories by Total Items Ordered', loc='left')
ax.invert_yaxis()
sns.despine(ax=ax, left=True, bottom=True)
ax.set_xticks([])

# Subplot 2: Worst 5 Product Categories by Total Items Ordered
ax = axes[0, 1]
bars = ax.barh(worst_items_ordered['product_category_name_english'],
               worst_items_ordered['total_items_ordered'],
               color='#00008B')
for bar in bars:
    width = bar.get_width()
    label_y = bar.get_y() + bar.get_height()/2
    ax.text(width/2, label_y, f'{int(width): ,}', ha='center', va='center', color='white')
ax.set_xlabel('Total Items Ordered')
ax.set_title('Worst 5 Product Categories by Total Items Ordered', loc='left')
ax.invert_yaxis()
sns.despine(ax=ax, left=True, bottom=True)
ax.set_xticks([])

# Subplot 3: Best 5 Product Categories by Total Order Value
ax = axes[1, 0]
bars = ax.barh(best_order_value['product_category_name_english'],
               best_order_value['total_order_value'],
               color='#00008B')
for bar in bars:
    width = bar.get_width()
    label_y = bar.get_y() + bar.get_height()/2
    ax.text(width/2, label_y, f'{int(width): ,}', ha='center', va='center', color='white')
ax.set_xlabel('Total Order Value (R$)')
ax.set_title('Best 5 Product Categories by Total Order Value', loc='left')
ax.invert_yaxis()
sns.despine(ax=ax, left=True, bottom=True)
ax.set_xticks([])

# Subplot 4: Worst 5 Product Categories by Total Order Value
ax = axes[1, 1]
bars = ax.barh(worst_order_value['product_category_name_english'],
               worst_order_value['total_order_value'],
               color='#00008B')
for bar in bars:
    width = bar.get_width()
    label_y = bar.get_y() + bar.get_height()/2
    ax.text(width/2, label_y, f'{int(width): ,}', ha='center', va='center', color='white')
ax.set_xlabel('Total Order Value (R$)')
ax.set_title('Worst 5 Product Categories by Total Order Value', loc='left')
ax.invert_yaxis()
sns.despine(ax=ax, left=True, bottom=True)
ax.set_xticks([])

# Subplot 5: Best 10 Product Categories by Items per Order
ax = axes[2, 0]
bars = ax.barh(best_items_per_order['product_category_name_english'],
               best_items_per_order['items_per_order'],
               color='#00008B')
for bar in bars:
    width = bar.get_width()
    label_y = bar.get_y() + bar.get_height()/2
    ax.text(width/2, label_y, f'{width:.2f}', ha='center', va='center', color='white')
ax.set_xlabel('Avg. Items per Order')
ax.set_title('Best 10 Product Categories by Items per Order', loc='left')
ax.invert_yaxis()
sns.despine(ax=ax, left=True, bottom=True)
ax.set_xticks([])

# Subplot 6: Best 10 Product Categories Pairs
ax = axes[2, 1]
bars = ax.barh(top_10_product_category_pairs['product_category_pair'],
               top_10_product_category_pairs['total_orders'],
               color='#00008B')
for bar in bars:
    width = bar.get_width()
    label_y = bar.get_y() + bar.get_height()/2
    ax.text(width/2, label_y, f'{int(width): ,}', ha='center', va='center', color='white')
ax.set_xlabel('Total Orders')
ax.set_title('Best 10 Product Categories Pairs', loc='left')
ax.invert_yaxis()
sns.despine(ax=ax, left=True, bottom=True)
ax.set_xticks([])

plt.tight_layout()
st.pyplot(fig3)

# --- Visualization 6: Delivery Time Performance ---
st.subheader("Delivery Time Performance")

# Data Loading Function for Delivered Orders
@st.cache_data
def load_orders_delivered():
    file_id = '18vmpeSFAaQiT8cLGbHbW-R4DHC1SS3iK'
    url = f'https://drive.google.com/uc?id={file_id}'
    output = 'orders_delivered_datasets.csv'
    gdown.download(url, output, quiet=False)
    df_delivered = pd.read_csv(output)
    # Convert purchase timestamp to datetime if available
    if 'order_purchase_timestamp' in df_delivered.columns:
        df_delivered['order_purchase_timestamp'] = pd.to_datetime(df_delivered['order_purchase_timestamp'])
    return df_delivered

# Data Loading Function for Brazil States Shapefile
@st.cache_data
def load_brazil_states():
    file_id = '1zoHAj7jNM9pj7yyfn4b4mEBdAAzAvp7n'
    url = f'https://drive.google.com/uc?id={file_id}'
    output = 'brazil_states.zip'
    gdown.download(url, output, quiet=False)
    brazil_states = gpd.read_file(output)
    return brazil_states

# --- Load Delivered Orders and Brazil States Datasets ---
df_orders_delivered = load_orders_delivered()
brazil_states = load_brazil_states()

# --- Filter Delivered Orders by Selected Date Range ---
df_orders_delivered_filtered = df_orders_delivered[
    (df_orders_delivered['order_purchase_timestamp'] >= pd.to_datetime(start_date)) &
    (df_orders_delivered['order_purchase_timestamp'] <= pd.to_datetime(end_date))
].copy()

# --- Create Customer Geolocation Table ---
customer_geolocation = df_orders_delivered_filtered.groupby('customer_zip_code_prefix').agg(
    customer_lat=('customer_lat', 'mean'),
    customer_lng=('customer_lng', 'mean'),
    customer_city=('customer_city', 'first'),
    customer_state=('customer_state', 'first'),
    median_delivery_time=('delivery_time', 'median')
).reset_index()

# --- Create State Delivery Table ---
state_delivery = df_orders_delivered_filtered.groupby('customer_state').agg(
    customer_lat=('customer_lat', 'mean'),
    customer_lng=('customer_lng', 'mean'),
    median_delivery_time=('delivery_time', 'median')
).reset_index()

# --- Determine Fastest and Slowest Delivery States ---
shortest_delivery_states = state_delivery.nsmallest(10, 'median_delivery_time')
longest_delivery_states = state_delivery.nlargest(10, 'median_delivery_time')

# --- Clean Brazil States Shapefile State Names ---
brazil_states['customer_state'] = brazil_states['HASC_1'].str.replace('BR.', '')

# --- Merge Shapefile with State Delivery Data ---
merged_data = brazil_states.merge(
    state_delivery,
    on='customer_state',
    how='left'
)

# --- Create Composite Figure for Delivery Time Performance ---
fig4 = plt.figure(figsize=(12, 12))
gs = fig4.add_gridspec(nrows=2, ncols=2, height_ratios=[4, 8], width_ratios=[6, 6])

# Subplot 1: 10 Fastest Delivery States
ax1 = fig4.add_subplot(gs[0, 0])
bars1 = ax1.barh(shortest_delivery_states['customer_state'],
                 shortest_delivery_states['median_delivery_time'],
                 color='#00008B')
for bar in bars1:
    width = bar.get_width()
    label_y = bar.get_y() + bar.get_height() / 2
    ax1.text(width/2, label_y, f'{width:.3f}', ha='center', va='center', color='white')
ax1.set_xlabel('Median Delivery Time (days)')
ax1.set_title('10 Fastest Delivery States', loc='center')
ax1.invert_yaxis()
sns.despine(ax=ax1, left=True, bottom=True)
ax1.set_xticks([])

# Subplot 2: 10 Slowest Delivery States
ax2 = fig4.add_subplot(gs[0, 1])
bars2 = ax2.barh(longest_delivery_states['customer_state'],
                 longest_delivery_states['median_delivery_time'],
                 color='#00008B')
for bar in bars2:
    width = bar.get_width()
    label_y = bar.get_y() + bar.get_height() / 2
    ax2.text(width/2, label_y, f'{width:.3f}', ha='center', va='center', color='white')
ax2.set_xlabel('Median Delivery Time (days)')
ax2.set_title('10 Slowest Delivery States', loc='center')
ax2.invert_yaxis()
sns.despine(ax=ax2, left=True, bottom=True)
ax2.set_xticks([])

# Subplot 3: Brazil State Map by Median Delivery Time
ax3 = fig4.add_subplot(gs[1, :])
merged_data.plot(
    column='median_delivery_time',
    cmap='viridis',
    linewidth=0.8,
    ax=ax3,
    edgecolor='0.8',
    legend=True
)
for _, row in merged_data.iterrows():
    centroid = row['geometry'].centroid
    ax3.text(centroid.x, centroid.y, row['customer_state'],
             fontsize=8, ha='center', color='black', weight='bold')
ax3.set_title('Brazil State Map by Median Delivery Time (days)', fontsize=12)
ax3.set_axis_off()

plt.tight_layout()
st.pyplot(fig4)
