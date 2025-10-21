import pandas as pd
import streamlit as st
import plotly.express as px
from sklearn.linear_model import LinearRegression
from datetime import timedelta
import numpy as np

# Set Streamlit page config
st.set_page_config(layout="wide", page_title="FDR Deposit Static Dashboard (Final)")

# Plotly Configuration for STATIC/NON-INTERACTIVE Plots (Toolbar OFF)
STATIC_CONFIG = {
    'displayModeBar': False,  # Hides the toolbar
    'doubleClick': False      # Disables the zoom on double-click
}

# Global Variables for consistent styling and formatting
FULL_NUMBER_FORMAT = ',.0f'
SHADOW_STYLE = "2px 2px 8px rgba(0,0,0,0.2)" # Consistent and visible shadow

# Function to load and clean data
@st.cache_data
def load_data(file_path):
    """Loads, cleans, and prepares the FDR data."""
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        st.error(f"Error: The file '{file_path}' was not found. Please ensure it's in the same directory.")
        return pd.DataFrame()

    # 1. Clean Column Names
    df.columns = df.columns.str.strip()
    
    # 2. Convert Data Types and Handle Missing Values
    df['Deposited_Amount'] = pd.to_numeric(df['Deposited_Amount'], errors='coerce')
    # Dropna requires both date columns for maturity/opening plots
    df.dropna(subset=['Deposited_Amount', 'FDR_Opening_Date', 'Mature_Date'], inplace=True)
    
    # Convert dates to datetime objects
    date_cols = ['FDR_Opening_Date', 'Mature_Date']
    for col in date_cols:
        try:
            df[col] = pd.to_datetime(df[col], format='%d-%m-%y', errors='coerce')
        except:
            df[col] = pd.to_datetime(df[col], errors='coerce')

    df.dropna(subset=['FDR_Opening_Date', 'Mature_Date'], inplace=True)
    
    if 'FDR_Current Status' in df.columns:
        df['FDR_Current Status'] = df['FDR_Current Status'].astype(str).str.strip()
    
    if 'Saving_Cycle' in df.columns:
        df['Saving_Cycle'] = df['Saving_Cycle'].astype(str).str.strip()

    return df

# Helper function to wrap charts in shadow box
def display_chart_with_shadow(fig, chart_title, use_container_width=True, config=STATIC_CONFIG):
    """Wraps st.plotly_chart in HTML div to apply shadow box style."""
    st.subheader(chart_title)
    st.markdown(f'<div style="box-shadow: {SHADOW_STYLE}; border-radius: 8px; padding: 5px; margin-bottom: 20px;">', unsafe_allow_html=True)
    st.plotly_chart(fig, use_container_width=use_container_width, config=config)
    st.markdown('</div>', unsafe_allow_html=True)

# --- 1. Load Data ---
file_path = "FDR_Sep_2025.csv"
df_full = load_data(file_path)

if df_full.empty:
    st.stop() 

st.title("✨ SEP 2025 Static Saving FDR Dashboard")
st.markdown("All **4 KPIs** and **all visualizations** are static and styled with shadows, showing deposited amounts in full numbers.")
st.markdown("---")

# --- 1st: Styled KPI (Key Performance Indicator) ---
st.header("Key Performance Indicators (KPIs)")

# Calculate KPIs on the full dataset
total_deposited_amount = df_full['Deposited_Amount'].sum()
average_deposit = df_full['Deposited_Amount'].mean()
status_counts = df_full['FDR_Current Status'].value_counts()
total_active_accounts = status_counts.get('Active', 0)
total_accounts = df_full.shape[0] # Total number of records

# Using 4 columns for the KPIs
col1, col2, col3, col4 = st.columns(4)

# KPI 1: Total Deposits
with col1:
    st.markdown(f"""
        <div style="background-color:#E8F8F5; padding: 15px; border-left: 5px solid #1ABC9C; border-radius: 4px; box-shadow: {SHADOW_STYLE};">
            <h5 style="color:#1ABC9C; margin: 0;">Total Deposits 💰</h5>
            <h2 style="color:#000000; margin: 5px 0 0;">{total_deposited_amount:,.0f}</h2>
        </div>
    """, unsafe_allow_html=True)

# KPI 2: Avg. Deposit
with col2:
    st.markdown(f"""
        <div style="background-color:#F4F6F6; padding: 15px; border-left: 5px solid #3498DB; border-radius: 4px; box-shadow: {SHADOW_STYLE};">
            <h5 style="color:#3498DB; margin: 0;">Avg. Deposit 💸</h5>
            <h2 style="color:#000000; margin: 5px 0 0;">{average_deposit:,.0f}</h2>
        </div>
    """, unsafe_allow_html=True)

# KPI 3: Total Accounts
with col3:
    st.markdown(f"""
        <div style="background-color:#F9EBEA; padding: 15px; border-left: 5px solid #E74C3C; border-radius: 4px; box-shadow: {SHADOW_STYLE};">
            <h5 style="color:#E74C3C; margin: 0;">Total FDR Accounts 📝</h5>
            <h2 style="color:#000000; margin: 5px 0 0;">{total_accounts:,.0f}</h2>
        </div>
    """, unsafe_allow_html=True)
    
# KPI 4: Total Active Accounts
with col4:
    st.markdown(f"""
        <div style="background-color:#F5EEF8; padding: 15px; border-left: 5px solid #9B59B6; border-radius: 4px; box-shadow: {SHADOW_STYLE};">
            <h5 style="color:#9B59B6; margin: 0;">Total Active Accounts ✅</h5>
            <h2 style="color:#000000; margin: 5px 0 0;">{total_active_accounts:,.0f}</h2>
        </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# --- 2nd: All Column Visualization (Static Plots with Shadows) ---
st.header(" All Column Visualization (Static Plots)")

# Row 1: Zone (Bar Chart) and Domain (Bar Chart)
col_vis1, col_vis_domain = st.columns(2)

with col_vis1:
    df_zone = df_full.groupby('Zone')['Deposited_Amount'].sum().reset_index()
    df_zone = df_zone.sort_values('Deposited_Amount', ascending=True) 
    
    fig_zone = px.bar(
        df_zone,
        x='Deposited_Amount', 
        y='Zone',             
        orientation='h',      
        title="Top Total Deposits Grouped by Zone",
        labels={'Deposited_Amount': 'Total Deposited Amount', 'Zone': 'Zone'},
        color='Deposited_Amount', 
        color_continuous_scale=px.colors.sequential.Teal, 
        template='plotly_white'
    )
    fig_zone.update_layout(
        coloraxis_showscale=False, 
        yaxis={'categoryorder': 'total ascending'},
        xaxis={'tickformat': FULL_NUMBER_FORMAT} 
    )
    display_chart_with_shadow(fig_zone, "Deposits by Zone ")


with col_vis_domain:
    df_domain = df_full.groupby('Domain')['Deposited_Amount'].sum().reset_index()
    
    fig_domain = px.bar(
        df_domain.sort_values('Deposited_Amount', ascending=False),
        x='Domain',
        y='Deposited_Amount',
        title="Total Deposited Amount by Domain",
        color='Domain',
        color_discrete_sequence=px.colors.qualitative.Bold,
        template='plotly_white'
    )
    fig_domain.update_layout(
        yaxis={'tickformat': FULL_NUMBER_FORMAT} 
    )
    display_chart_with_shadow(fig_domain, "Deposits by Domain")


# Row 2: Product and Status
col_vis2, col_vis3 = st.columns(2)

with col_vis2:
    df_product = df_full.groupby('Saaving_Product')['Deposited_Amount'].sum().reset_index()
    fig_product = px.pie(
        df_product,
        names='Saaving_Product',
        values='Deposited_Amount',
        title="Deposited Amount Distribution by Saving Product (%)",
        template='plotly_white'
    )
    display_chart_with_shadow(fig_product, "Deposits by Saving Product")


with col_vis3:
    df_status = df_full['FDR_Current Status'].value_counts().reset_index()
    df_status.columns = ['Status', 'Count']
    fig_status = px.bar(
        df_status,
        x='Status',
        y='Count',
        title="Count of Accounts by FDR Status",
        color='Status',
        template='plotly_white'
    )
    display_chart_with_shadow(fig_status, "Accounts by Current Status")


# Row 3: Duration and Saving Cycle
col_vis4, col_vis5 = st.columns(2)

with col_vis4:
    df_duration = df_full.groupby('Duration')['Deposited_Amount'].sum().reset_index()
    fig_duration = px.bar(
        df_duration.sort_values('Deposited_Amount', ascending=False),
        x='Duration',
        y='Deposited_Amount',
        title="Total Deposits by Fixed Deposit Duration",
        color='Duration',
        template='plotly_white'
    )
    fig_duration.update_layout(
        yaxis={'tickformat': FULL_NUMBER_FORMAT} 
    )
    display_chart_with_shadow(fig_duration, "Deposits by Duration")

with col_vis5:
    df_cycle = df_full.groupby('Saving_Cycle')['Deposited_Amount'].sum().reset_index()
    fig_cycle = px.bar(
        df_cycle.sort_values('Deposited_Amount', ascending=False),
        x='Saving_Cycle',
        y='Deposited_Amount',
        title="Total Deposits by Saving Cycle",
        labels={'Deposited_Amount': 'Total Deposited Amount', 'Saving_Cycle': 'Saving Cycle'},
        color='Saving_Cycle',
        template='plotly_white'
    )
    fig_cycle.update_layout(
        yaxis={'tickformat': FULL_NUMBER_FORMAT} 
    )
    display_chart_with_shadow(fig_cycle, "Deposited Amount by Saving Cycle")

# Row 4 (Full Width)
# Opening Date (Line Chart) and Mature Date (Area Chart) combined into two subplots if space allowed, 
# but keeping them sequential for readability and consistent shadow wrapping.

# Opening Date Plot
df_date_deposit = df_full.groupby('FDR_Opening_Date')['Deposited_Amount'].sum().reset_index()
df_date_deposit = df_date_deposit.sort_values('FDR_Opening_Date')

fig_date_deposit = px.line(
    df_date_deposit,
    x='FDR_Opening_Date',
    y='Deposited_Amount',
    title="Total Daily Deposited Amount Over Time",
    labels={'Deposited_Amount': 'Total Deposited Amount', 'FDR_Opening_Date': 'Opening Date'},
    template='plotly_white'
)
fig_date_deposit.update_layout(
    yaxis={'tickformat': FULL_NUMBER_FORMAT} 
)
display_chart_with_shadow(fig_date_deposit, "Deposited Amount by Opening Date")

# NEW ROW: Mature Date (Area Chart - New Style)
df_mature_deposit = df_full.groupby('Mature_Date')['Deposited_Amount'].sum().reset_index()
df_mature_deposit = df_mature_deposit.sort_values('Mature_Date')

# Use Area Chart for visualization of maturity volume over time
fig_mature_deposit = px.area(
    df_mature_deposit,
    x='Mature_Date',
    y='Deposited_Amount',
    title="Total Maturity Amount Exposure Over Time",
    labels={'Deposited_Amount': 'Total Matured Amount', 'Mature_Date': 'Maturity Date'},
    line_shape='spline', # Smooth the line for better flow
    color_discrete_sequence=['#FF7F0E'], # Distinct color for the area
    template='plotly_white'
)
fig_mature_deposit.update_layout(
    yaxis={'tickformat': FULL_NUMBER_FORMAT} 
)
display_chart_with_shadow(fig_mature_deposit, "Deposited Amount by Mature Date")


st.markdown("---")

# --- 3rd: Prediction / Forecasting (Static with Shadow) ---
st.header("Deposited Amount Next Month Prediction")

# Aggregate data by date
df_daily = df_full.groupby('FDR_Opening_Date')['Deposited_Amount'].sum().reset_index()
df_daily.sort_values('FDR_Opening_Date', inplace=True)

if len(df_daily) > 1:
    # Feature Engineering and Linear Regression model training (unchanged)
    df_daily['Days_Since_Start'] = (df_daily['FDR_Opening_Date'] - df_daily['FDR_Opening_Date'].min()).dt.days
    X = df_daily[['Days_Since_Start']].values
    y = df_daily['Deposited_Amount'].values
    model = LinearRegression()
    model.fit(X, y)
    slope = model.coef_[0]
    mean_daily_deposit = y.mean()
    meaningful_threshold = mean_daily_deposit * 0.01 
    
    if slope > meaningful_threshold:
        trend_analysis = "INCREASE 📈"
        trend_color = "green"
    elif slope < -meaningful_threshold:
        trend_analysis = "DECREASE 📉"
        trend_color = "red"
    else:
        trend_analysis = "STABLE ➡️"
        trend_color = "blue"
        
    # Prediction logic 
    last_day = df_daily['FDR_Opening_Date'].max()
    last_day_index = df_daily['Days_Since_Start'].max()
    num_days_forecast = 30
    forecast_dates = [last_day + timedelta(days=i) for i in range(1, num_days_forecast + 1)]
    forecast_days_since_start = np.array([last_day_index + i for i in range(1, num_days_forecast + 1)]).reshape(-1, 1)
    predictions = model.predict(forecast_days_since_start)
    
    df_prediction = pd.DataFrame({
        'FDR_Opening_Date': forecast_dates,
        'Amount': np.maximum(0, predictions),
        'Type': 'Predicted'
    })

    df_historical_plot = df_daily.rename(columns={'Deposited_Amount': 'Amount'})
    df_historical_plot['Type'] = 'Actual'
    df_combined = pd.concat([df_historical_plot[['FDR_Opening_Date', 'Amount', 'Type']], 
                            df_prediction[['FDR_Opening_Date', 'Amount', 'Type']]])

    # Trend Output - Simple Statement
    st.subheader("Predicted Trend")
    st.markdown(f"Based on historical data, the **Deposited Amount** trend for the next month is predicted to **<span style='color:{trend_color}; font-size: 24px;'>{trend_analysis}</span>**.", unsafe_allow_html=True)
    
    # Visualization of the forecast
    st.subheader("Forecast Plot (Next 30 Days)")
    fig_forecast = px.line(
        df_combined,
        x='FDR_Opening_Date',
        y='Amount',
        color='Type',
        title='Daily Deposit Amount Forecast (Historical vs. Predicted)',
        labels={'Amount': 'Deposited Amount', 'FDR_Opening_Date': 'Date'},
        template='plotly_white'
    )
    fig_forecast.update_layout(
        yaxis={'tickformat': FULL_NUMBER_FORMAT} 
    )
    
    # Wrap chart in styled container
    st.markdown(f'<div style="box-shadow: {SHADOW_STYLE}; border-radius: 8px; padding: 5px;">', unsafe_allow_html=True)
    st.plotly_chart(fig_forecast, use_container_width=True, config=STATIC_CONFIG)
    st.markdown('</div>', unsafe_allow_html=True)

else:
    st.warning("Not enough daily data points (need at least 2) in the dataset to run a linear regression model for prediction.")
    
# --- End of Streamlit Dashboard Script ---
