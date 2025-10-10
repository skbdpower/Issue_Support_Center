import streamlit as st
import pandas as pd
import altair as alt
from statsmodels.tsa.statespace.sarimax import SARIMAX
import matplotlib.pyplot as plt
from datetime import date

# --- 1. CONFIGURATION & CUSTOM STYLING (Minimalist Forest Green Theme) ---
st.set_page_config(layout="wide", page_title="Loan Disbursement KPI Dashboard (Forest Green)", initial_sidebar_state="expanded")

# Inject Custom CSS for the Minimalist Theme (Font Sizes Adjusted for Smaller View)
st.markdown("""
<style>
/* --- THEME COLORS --- */
/* Backgrounds: #f7f7f7 (Light Gray/White) */
/* Card/Widget Background: #ffffff (White) */
/* Text: #222222 (Near Black) */
/* Accents: #1E4D2B (Forest Green) */
/* Metrics/Financial Highlight: #FFCC00 (Warm Gold/Mustard) */

/* 1. GLOBAL FONT AND STYLING (Smaller General Text) */
html, body, [class*="st-"] {
    font-family: 'Inter', 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
    color: #222222; 
    font-size: 0.8rem; /* Small General Text */
}
.main {
    background-color: #f7f7f7; 
}

/* 2. KPI BOX STYLING */
.st-emotion-cache-1r6ch5d, .st-emotion-cache-1cpx93x, .st-emotion-cache-1y4pm5l {
    background-color: #ffffff; 
    border-radius: 4px; 
    padding: 16px; 
    box-shadow: none; 
    border: 1px solid #e0e0e0; 
    margin-bottom: 15px;
}
/* Adjusting metric value and label size (Smaller Overall) */
.st-emotion-cache-1wv0k8c p, .st-emotion-cache-n69a00 p {
    font-size: 1.2rem; /* Small KPI Value */
    font-weight: 800;
    color: #1E4D2B; 
}
.st-emotion-cache-1wv0k8c div:first-child p {
    font-size: 0.6rem; /* Very Small KPI Label */
    color: #666666; 
    font-weight: 600;
    text-transform: uppercase;
}
/* Specific style for the PAR rate delta (Risk Highlight) */
.st-emotion-cache-1wv0k8c .st-emotion-cache-1d0o43a, .st-emotion-cache-n69a00 .st-emotion-cache-1d0o43a {
    color: #CC0000; /* Red for Delta/Risk amount */
}


/* 3. HEADER STYLING (Smaller Headers) */
h1 {
    color: #1E4D2B; 
    border-bottom: 2px solid #FFCC00; 
    padding-bottom: 10px;
    margin-bottom: 20px; 
    font-size: 1.2rem; /* Small H1 */
}
h2 {
    color: #1E4D2B;
    border-left: 5px solid #FFCC00; 
    padding-left: 10px;
    margin-top: 25px;
    margin-bottom: 15px;
    font-size: 0.8rem; /* Small H2 */
}
h3 {
    color: #1E4D2B; 
    font-size: 1.0rem; /* Small H3 */
}

/* 4. ALTAIR CHART CONTAINER */
.st-emotion-cache-1ftn3f8 {
    padding: 0; 
    background-color: #ffffff; 
    border-radius: 4px;
    border: 1px solid #e0e0e0;
}

/* 5. Custom Prediction Box Style */
.prediction-box {
    background-color: #ffffff;
    padding: 20px;
    border-radius: 4px;
    border-left: 6px solid #FFCC00; 
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
}
.prediction-box h3 {
    color: #1E4D2B !important; 
    margin-top: 0;
    font-size: 1.1rem; 
}
.trend-increased {
    color: #008000; 
    font-weight: 800;
    font-size: 1.0em; 
}
.trend-decreased {
    color: #CC0000; 
    font-weight: 800;
    font-size: 1.0em; 
}
</style>
""", unsafe_allow_html=True)

# Configure Matplotlib for the Light Theme
plt.style.use('default') 
plt.rcParams.update({
    'axes.titlecolor': '#1E4D2B',
    'axes.labelcolor': '#222222',
    'xtick.color': '#222222',
    'ytick.color': '#222222',
    'grid.color': '#e0e0e0',
    'figure.facecolor': '#f7f7f7',
    'axes.facecolor': '#ffffff',
})

st.title("🌱 ALL DOMAIN LOAN DISBURSEMENT & RISK REPORT SEP-2025")
st.markdown("---")


# --- 2. DATA LOADING & FILTERING ---
# --- FIX 1: Change file extension to XLSX ---
DATA_FILE = 'all_domain_loan_disburse_report_sep_2025.xlsx' 

@st.cache_data
def load_data(file_path):
    # --- FIX 2: Change to pd.read_excel for .xlsx files ---
    try:
        df = pd.read_excel(file_path)
    except Exception as e:
        # Handle case where file might still be CSV, or the Excel library is missing
        st.error(f"Error reading Excel file. Make sure the file exists and you have 'openpyxl' installed (pip install openpyxl). Underlying error: {e}")
        raise e

    # Date Format Fix
    date_cols = ['Disbursment_Date', 'Admission_Date', 'First_Repayment_Date']
    for col in date_cols:
        # Excel typically handles dates better, but we ensure consistency
        df[col] = pd.to_datetime(df[col], errors='coerce') 

    # Convert relevant columns to numeric
    numeric_cols = ['Loan_amount', 'Insurance_Amount', 'Outstanding_Pr', 'Due_Amount_Pr', 'Age', 'No_Of_Installment', 'Sc_Rate'] 
    
    # --- FIX APPLIED HERE ---
    for col in numeric_cols:
         # 1. Handle percentage strings if present
         if df[col].dtype == object and '%' in df[col].astype(str).str.cat(sep=''):
            df[col] = df[col].astype(str).str.replace('%', '', regex=False)
            
         # 2. Convert to numeric and fill NaNs with 0
         df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0) 

    # Ensure integer types where appropriate
    df['Age'] = df['Age'].astype(int, errors='ignore')
    df['No_Of_Installment'] = df['No_Of_Installment'].astype(int, errors='ignore')
    
    # NEW: Create a risk indicator column (Portfolio at Risk: Due amount > 0)
    df['Is_At_Risk'] = df['Due_Amount_Pr'] > 0
    
    return df

try:
    df_raw = load_data(DATA_FILE) 
    
    # Determine the overall date range for the sidebar
    min_date_raw = df_raw['Disbursment_Date'].min().date() if not df_raw['Disbursment_Date'].empty and not pd.isna(df_raw['Disbursment_Date'].min()) else date(2025, 9, 1)
    max_date_raw = df_raw['Disbursment_Date'].max().date() if not df_raw['Disbursment_Date'].empty and not pd.isna(df_raw['Disbursment_Date'].max()) else date(2025, 9, 30)

except Exception as e:
    # This catch handles errors from load_data
    st.error(f"Data loading and processing error: {e}")
    st.stop()


# --- 2.1 SIDEBAR FILTERING (New Interactive Component) ---
st.sidebar.header("Date Range Filter")
start_date = st.sidebar.date_input("Start Date", min_date_raw, min_value=min_date_raw, max_value=max_date_raw)
end_date = st.sidebar.date_input("End Date", max_date_raw, min_value=min_date_raw, max_value=max_date_raw)

# Apply the filter
filtered_df = df_raw[
    (df_raw['Disbursment_Date'].dt.date >= start_date) & 
    (df_raw['Disbursment_Date'].dt.date <= end_date)
].copy()

if filtered_df.empty:
    st.warning("No data available for the selected date range. Please adjust the filter.")
    st.stop()
    
df = filtered_df # Use the filtered DataFrame for all subsequent calculations


# --- ALTAIR THEME CONFIGURATION ---
altair_forest_theme = {
    "config": {
        "title": {"color": "#1E4D2B", "fontSize": 16, "font": 'Inter'},
        "axis": {
            "titleColor": "#666666",
            "labelColor": "#222222",
            "gridColor": "#e0e0e0",
            "domainColor": "#cccccc",
            "labelFont": 'Inter',
            "titleFont": 'Inter'
        },
        "header": {"titleColor": "#222222", "labelColor": "#666666"},
        "legend": {"titleColor": "#222222", "labelColor": "#666666"},
        "range": {
            # Forest, Gold, and Earthy Tones
            "category": ['#1E4D2B', '#FFCC00', '#556B2F', '#CC5500', '#A9A9A9', '#708090', '#6B8E23', '#4682B4']
        }
    }
}
alt.themes.register("altair_forest_theme", lambda: altair_forest_theme)
alt.themes.enable("altair_forest_theme")


# --- 3. TOP-LEVEL KPIS (STYLED METRICS) ---
st.header(f"KPI Summary: Volume and Performance ({start_date} to {end_date})")

# Calculate KPIs on the filtered_df
total_loan_amount = df['Loan_amount'].sum()
total_insurance_amount = df['Insurance_Amount'].sum()
total_outstanding_pr = df['Outstanding_Pr'].sum()
total_due_amount_pr = df['Due_Amount_Pr'].sum()
total_borrowers = df['BorrowerCode'].nunique()

# Performance KPIs
# NEW: Portfolio at Risk (PAR) 
at_risk_loan_amount = df[df['Is_At_Risk']]['Loan_amount'].sum()
par_rate = (at_risk_loan_amount / total_loan_amount) if total_loan_amount > 0 else 0

# Insurance Coverage Ratio
insurance_coverage_ratio = (total_insurance_amount / total_loan_amount) if total_loan_amount > 0 else 0


col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric("Total Loan Amount Disbursed", f"{total_loan_amount:,.0f}")
with col2:
    st.metric("Total Outstanding Principal", f"{total_outstanding_pr:,.0f}")
with col3:
    # HIGHLIGHTED NEW METRIC
    st.metric("Portfolio At Risk (PAR Rate)", f"{par_rate:.2%}", delta=f"Risk Amt: {at_risk_loan_amount:,.0f}") 
with col4:
    st.metric("Total Unique Borrowers", f"{total_borrowers:,.0f}")
with col5:
    st.metric("Insurance Coverage Ratio", f"{insurance_coverage_ratio:.2%}")

st.markdown("---")


# --- 4. NEW: LOAN RISK CONCENTRATION ANALYSIS ---

st.header("Loan Performance & Risk Concentration (PAR Rate)")

col_risk_left, col_risk_right = st.columns(2)

# --- Plot A: PAR Rate by Divisional Office ---
df_risk_div = df.groupby('Divisional_Office', as_index=False).agg(
    Total_Loan=('Loan_amount', 'sum'),
    Risk_Loan=('Is_At_Risk', lambda x: df.loc[x.index, 'Loan_amount'].where(x).sum())
).assign(
    PAR_Rate=lambda x: x['Risk_Loan'] / x['Total_Loan']
).sort_values('PAR_Rate', ascending=False)

with col_risk_left:
    st.markdown("### Portfolio At Risk Rate by Divisional Office")

    # Dynamic domain for color scale
    par_min = df_risk_div['PAR_Rate'].min()
    par_mean = df_risk_div['PAR_Rate'].mean()
    par_max = df_risk_div['PAR_Rate'].max()
    
    # Bar chart colored by PAR_Rate (Red is high risk, Green is low risk)
    chart_risk_div = alt.Chart(df_risk_div).mark_bar().encode(
        x=alt.X('PAR_Rate', title='PAR Rate', axis=alt.Axis(format='%')),
        y=alt.Y('Divisional_Office', sort='-x', title='Divisional Office'),
        # Color scale: Red for high PAR, Green for low PAR
        color=alt.Color('PAR_Rate', 
                        scale=alt.Scale(range=['#008000', '#FFCC00', '#CC0000'], domain=[par_min, par_mean, par_max]),
                        legend=alt.Legend(title="PAR Rate")),
        tooltip=['Divisional_Office', alt.Tooltip('PAR_Rate', format='.2%'), alt.Tooltip('Risk_Loan', format=',.0f', title='At Risk Loan Amt')]
    ).properties(
        title='Divisional Risk Performance'
    )
    st.altair_chart(chart_risk_div, use_container_width=True)


# --- Plot B: PAR Rate by Current Loan Product ---
df_risk_product = df.groupby('Current_Loan_Product', as_index=False).agg(
    Total_Loan=('Loan_amount', 'sum'),
    Risk_Loan=('Is_At_Risk', lambda x: df.loc[x.index, 'Loan_amount'].where(x).sum())
).assign(
    PAR_Rate=lambda x: x['Risk_Loan'] / x['Total_Loan']
).sort_values('PAR_Rate', ascending=False)

with col_risk_right:
    st.markdown("### Portfolio At Risk Rate by Loan Product")
    
    chart_risk_product = alt.Chart(df_risk_product).mark_bar(color='#CC5500').encode( # Deep Orange/Rust color for risk
        x=alt.X('PAR_Rate', title='PAR Rate', axis=alt.Axis(format='%')),
        y=alt.Y('Current_Loan_Product', sort='-x', title='Loan Product'),
        tooltip=['Current_Loan_Product', alt.Tooltip('PAR_Rate', format='.2%'), alt.Tooltip('Risk_Loan', format=',.0f', title='At Risk Loan Amt')]
    ).properties(
        title='Product Risk Performance'
    )
    st.altair_chart(chart_risk_product, use_container_width=True)

st.markdown("---")


# --- 5. GEOGRAPHIC LOAN DISTRIBUTION (Original Section 4) ---

st.header("Geographic Loan Distribution: Divisional & Zone Offices")

col_geo_left, col_geo_right = st.columns(2)

# Data preparation for Divisional Office
df_div = df.groupby('Divisional_Office', as_index=False).agg(
    Total_Loan_Amount=('Loan_amount', 'sum')
).sort_values('Total_Loan_Amount', ascending=False)
# Calculate percentages
total_sum = df_div['Total_Loan_Amount'].sum()
df_div['Percentage'] = df_div['Total_Loan_Amount'] / total_sum

with col_geo_left:
    # --- Plot 1: Divisional Office Wise Plot (ENHANCED BAR CHART) ---
    st.markdown("### Total Loan Amount by Divisional Office")

    # Base Bar Chart (Using Forest Green color)
    base_bar = alt.Chart(df_div).mark_bar(color='#1E4D2B').encode(
        x=alt.X('Total_Loan_Amount', title='Total Loan'),
        y=alt.Y('Divisional_Office', sort='-x', title='Divisional Office'),
        # Color based on value magnitude for visual impact
        color=alt.Color('Total_Loan_Amount', scale=alt.Scale(range='ramp'), legend=None), 
        tooltip=[
            'Divisional_Office', 
            alt.Tooltip('Total_Loan_Amount', format=',.0f', title='Total Loan'),
            alt.Tooltip('Percentage', format='.1%', title='Share')
        ]
    ).properties(
        title='Divisional Office Performance by Loan Amount'
    )
    
    # Text Layer to show Percentage
    text_layer = base_bar.mark_text(
        align='left',
        baseline='middle',
        dx=3 
    ).encode(
        x=alt.X('Total_Loan_Amount'),
        y=alt.Y('Divisional_Office', sort='-x'),
        text=alt.Text('Percentage', format='.1%'),
        color=alt.value("#222222") 
    )
    
    # Combine Bar and Text
    chart_div_enhanced = base_bar + text_layer

    st.altair_chart(chart_div_enhanced, use_container_width=True)


with col_geo_right:
    # --- Plot 2: Zone Office Wise Plot (Outstanding Principal) ---
    df_zone = df.groupby(['Divisional_Office', 'Zone_Office'], as_index=False).agg(
        Total_Outstanding_Pr=('Outstanding_Pr', 'sum')
    ).sort_values('Total_Outstanding_Pr', ascending=False).head(15) 

    chart_zone = alt.Chart(df_zone).mark_bar().encode(
        x=alt.X('Total_Outstanding_Pr', title='Total Outstanding Pr.'),
        y=alt.Y('Zone_Office', sort='-x', title='Zone Office'),
        # Use Division to color-code with the new palette
        color=alt.Color('Divisional_Office', legend=alt.Legend(title="Division")),
        tooltip=['Divisional_Office', 'Zone_Office', alt.Tooltip('Total_Outstanding_Pr', format=',.0f')]
    ).properties(
        title='Top Zone Offices by Outstanding Principal'
    )
    st.altair_chart(chart_zone, use_container_width=True)


# --- 6. DETAILED LOAN DISTRIBUTION ANALYSIS (2x3 Grid Layout) (Original Section 5) ---

st.markdown("---")
st.header("Detailed Loan Distribution Analysis")

# First row of 3 plots
col_r1_1, col_r1_2, col_r1_3 = st.columns(3)

# -------------------------
# Plot 3 (Row 1, Col 1): Cycle by Loan_amount (Total)
# -------------------------
df_cycle = df.groupby('Cycle', as_index=False).agg(
    Total_Loan=('Loan_amount', 'sum')
)
# Gold color
chart_cycle = alt.Chart(df_cycle).mark_bar(color='#FFCC00').encode(
    x=alt.X('Cycle', title='Loan Cycle', type='nominal'),
    y=alt.Y('Total_Loan', title='Total Loan Amount'),
    tooltip=['Cycle', alt.Tooltip('Total_Loan', format=',.0f')]
).properties(title='Total Loan Amount by Loan Cycle')
with col_r1_1:
    st.altair_chart(chart_cycle, use_container_width=True)


# -------------------------
# Plot 4 (Row 1, Col 2): Purpose by Loan_amount (Total, Top 10)
# -------------------------
df_purpose = df.groupby('Purpose', as_index=False).agg(
    Avg_Loan=('Loan_amount', 'mean'),
    Total_Loan=('Loan_amount', 'sum')
).sort_values('Total_Loan', ascending=False).head(10)
# Forest Green color
chart_purpose = alt.Chart(df_purpose).mark_bar(color='#1E4D2B').encode( 
    x=alt.X('Total_Loan', title='Total Loan Amount'),
    y=alt.Y('Purpose', sort='-x', title='Loan Purpose'),
    tooltip=['Purpose', alt.Tooltip('Total_Loan', format=',.0f'), alt.Tooltip('Avg_Loan', format=',.0f')]
).properties(title='Top 10 Purposes by Total Loan Amount')
with col_r1_2:
    st.altair_chart(chart_purpose, use_container_width=True)


# -------------------------
# Plot 5 (Row 1, Col 3): Frequency by Loan Amount (Total)
# -------------------------
df_frequency = df.groupby('Frequency', as_index=False).agg(
    Total_Loan=('Loan_amount', 'sum')
).sort_values('Total_Loan', ascending=False)
chart_frequency = alt.Chart(df_frequency).mark_bar().encode(
    x=alt.X('Total_Loan', title='Total Loan Amount'),
    y=alt.Y('Frequency', sort='-x', title='Repayment Frequency'),
    color=alt.Color('Frequency', legend=None), 
    tooltip=['Frequency', alt.Tooltip('Total_Loan', format=',.0f')]
).properties(title='Total Loan Amount by Repayment Frequency')
with col_r1_3:
    st.altair_chart(chart_frequency, use_container_width=True)


# Second row of 3 plots
col_r2_1, col_r2_2, col_r2_3 = st.columns(3)

# -------------------------
# Plot 6 (Row 2, Col 1): Borrower Age Histogram
# -------------------------
# Deep Orange color (Accent from palette)
chart_age_hist = alt.Chart(df).mark_bar(color='#CC5500').encode( 
    x=alt.X('Age', bin=alt.Bin(maxbins=15), title='Borrower Age Group'),
    y=alt.Y('count()', title='Count of Loans/Borrowers'),
    tooltip=[alt.Tooltip('Age', bin=True, title='Age Range'), 'count()']
).properties(title='Distribution of Loan Count by Borrower Age')
with col_r2_1:
    st.altair_chart(chart_age_hist, use_container_width=True)


# -------------------------
# Plot 7 (Row 2, Col 2): No_Of_Installment by Loan Amount (Total) - NEW PLOT
# -------------------------
df_installments = df.groupby('No_Of_Installment', as_index=False).agg(
    Total_Loan=('Loan_amount', 'sum')
).sort_values('No_Of_Installment', ascending=False).head(15) # Show top 15 installment options

chart_installments = alt.Chart(df_installments).mark_bar(color='#708090').encode( # Slate Gray
    x=alt.X('Total_Loan', title='Total Loan Amount'),
    y=alt.Y('No_Of_Installment', sort='-x', title='No. of Installments', type='nominal'),
    tooltip=['No_Of_Installment', alt.Tooltip('Total_Loan', format=',.0f')]
).properties(title='Total Loan Amount by Number of Installments')
with col_r2_2:
    st.altair_chart(chart_installments, use_container_width=True)


# -------------------------
# Plot 8 (Row 2, Col 3): Current_Loan_Product by Loan_amount
# -------------------------
# This data is already calculated in the Risk section, but we'll recalculate here for clarity
df_product = df.groupby('Current_Loan_Product', as_index=False).agg(
    Total_Loan=('Loan_amount', 'sum')
).sort_values('Total_Loan', ascending=False)

# Earthy Green color
chart_product = alt.Chart(df_product).mark_bar(color='#556B2F').encode( 
    x=alt.X('Total_Loan', title='Total Loan Amount'),
    y=alt.Y('Current_Loan_Product', sort='-x', title='Loan Product'),
    tooltip=['Current_Loan_Product', alt.Tooltip('Total_Loan', format=',.0f')]
).properties(title='Total Loan Amount by Current Loan Product')
with col_r2_3:
    st.altair_chart(chart_product, use_container_width=True)

st.markdown("---")


# --- 7. TIME SERIES ANALYSIS (2-COLUMN LAYOUT) (Original Section 8) ---

st.header("Time Series Analysis: Acquisition vs. Disbursement Trend (Daily)")

col_time_left, col_time_right = st.columns(2)

# -------------------------
# Plot 13 (Left): Admission_Date by BorrowerCode (Acquisition Trend - Daily)
# -------------------------
df_acq = df.groupby('Admission_Date', as_index=False).agg(
    New_Borrowers=('BorrowerCode', 'nunique')
).set_index('Admission_Date').resample('D').sum().reset_index().sort_values('Admission_Date')

chart_acq = alt.Chart(df_acq).mark_line(point=True, color='#556B2F', strokeWidth=3).encode( # Earthy Green
    x=alt.X('Admission_Date', title='Admission Date', axis=alt.Axis(format="%Y-%m-%d")),
    y=alt.Y('New_Borrowers', title='Daily New Borrower Count'),
    tooltip=[alt.Tooltip('Admission_Date', title='Date', format='%Y-%m-%d'), 'New_Borrowers']
).properties(
    title='Daily New Client Acquisition Trend'
) 

with col_time_left:
    st.altair_chart(chart_acq, use_container_width=True)

# -------------------------
# Plot 14 (Right): Disbursement_Date by Loan Amount (Disbursement Trend - Daily) - ENHANCED WITH ROLLING AVG
# -------------------------
df_time = df.groupby('Disbursment_Date', as_index=False).agg(
    Daily_Total_Loan=('Loan_amount', 'sum')
).set_index('Disbursment_Date').resample('D').sum().reset_index().sort_values('Disbursment_Date')

# NEW: Calculate 7-Day Rolling Average
df_time['7_Day_Avg'] = df_time['Daily_Total_Loan'].rolling(window=7, min_periods=1).mean().fillna(0)


# Base line chart (Daily Total Loan)
chart_line = alt.Chart(df_time).mark_line(point=True, color='#1E4D2B', strokeWidth=2).encode( # Forest Green
    x=alt.X('Disbursment_Date', title='Disbursement Date', axis=alt.Axis(format="%Y-%m-%d")), 
    y=alt.Y('Daily_Total_Loan', title='Daily Total Loan Amount'),
    tooltip=[alt.Tooltip('Disbursment_Date', title='Date', format='%Y-%m-%d'), alt.Tooltip('Daily_Total_Loan', format=',.0f', title='Daily Loan')]
)

# Rolling Average line chart
chart_avg = alt.Chart(df_time).mark_line(color='#FFCC00', strokeDash=[5, 5], strokeWidth=3).encode( # Gold - Dashed Line
    x='Disbursment_Date', 
    y=alt.Y('7_Day_Avg', title='Daily Total Loan Amount'),
    tooltip=[alt.Tooltip('7_Day_Avg', format=',.0f', title='7-Day Avg')]
)

# Combine the two charts
chart_time_enhanced = (chart_line + chart_avg).properties(
    title='Daily Total Loan Disbursement Trend (w/ 7-Day Rolling Avg)'
)

with col_time_right:
    st.altair_chart(chart_time_enhanced, use_container_width=True)

st.markdown("---")

# --- 9. Demographic and Micro-Geographic Analysis (RESTRUCTURED) ---

st.markdown("---")
st.header("Demographic and Micro-Geographic Analysis")

# First Row (Borrower and Gender)
col_borrower, col_gender = st.columns(2)

with col_borrower:
    # -------------------------
    # Plot 19: BorrowerCode by Loan Amount (TOP 25 Borrowers) --- HEIGHT INCREASED ---
    # -------------------------
    st.markdown("### Top 25 Borrowers by Total Loan Amount")

    # Group by BorrowerCode and sum loan amounts
    df_borrower = df.groupby('BorrowerCode', as_index=False).agg(
        Total_Loan=('Loan_amount', 'sum')
    ).sort_values('Total_Loan', ascending=False).head(25)

    # Use a distinguishing color (Warm Gold/Mustard)
    chart_borrower = alt.Chart(df_borrower).mark_bar(color='#FFCC00').encode(
        x=alt.X('Total_Loan', title='Total Loan Amount'),
        y=alt.Y('BorrowerCode', sort='-x', title='Borrower Code (Top 25)'),
        tooltip=['BorrowerCode', alt.Tooltip('Total_Loan', format=',.0f')]
    ).properties(
        title='Top 25 Clients by Portfolio Share',
        height=650 # INCREASED HEIGHT FOR BETTER VISIBILITY
    )
    st.altair_chart(chart_borrower, use_container_width=True)

with col_gender:
    # -------------------------
    # Plot 15: Gender by Loan Amount (Total)
    # -------------------------
    st.markdown("### Total Loan Amount by Gender")
    df_gender = df.groupby('Gender', as_index=False).agg(
        Total_Loan=('Loan_amount', 'sum'),
        Borrower_Count=('BorrowerCode', 'nunique')
    ).sort_values('Total_Loan', ascending=False)
    chart_gender = alt.Chart(df_gender).mark_bar().encode(
        x=alt.X('Total_Loan', title='Total Loan Amount'),
        y=alt.Y('Gender', sort='-x', title='Gender'),
        color=alt.Color('Gender', legend=None),
        tooltip=['Gender', alt.Tooltip('Total_Loan', format=',.0f'), 'Borrower_Count']
    ).properties(
        title='Total Loan Amount by Gender',
        height=650 # Increased height to match Borrower chart
    )
    st.altair_chart(chart_gender, use_container_width=True)


# Second Row (Village) - Branch is moved to the next section
col_village = st.columns(1)[0]

with col_village:
    # -------------------------
    # Plot 16: Village by Loan Amount (Top 25) 
    # -------------------------
    st.markdown("### Top 25 Villages by Loan Volume") 

    df_village = df.groupby('Village', as_index=False).agg(
        Total_Loan=('Loan_amount', 'sum'),
        Borrower_Count=('BorrowerCode', 'nunique')
    ).sort_values('Total_Loan', ascending=False).head(25)

    # Slate Blue color
    chart_village = alt.Chart(df_village).mark_bar(color='#4682B4').encode( 
        x=alt.X('Total_Loan', title='Total Loan Amount'),
        y=alt.Y('Village', sort='-x', title='Village (Top 25)', axis=alt.Axis(labelLimit=300)), 
        tooltip=['Village', alt.Tooltip('Total_Loan', format=',.0f'), 'Borrower_Count']
    ).properties(
        title='Top 25 Villages by Total Loan Amount',
        height=650 
    )
    st.altair_chart(chart_village, use_container_width=True)


# --- 10. BRANCH AND SUCCESS RATE ANALYSIS (Original Sections 7, 9) ---
st.markdown("---")
st.header("Branch and Success Rate Analysis")

col_b_left, col_b_right = st.columns(2)

with col_b_left:
    # -------------------------
    # Plot 17: Branch_Name by Loan Amount (Top 25) 
    # -------------------------
    st.markdown("### Top 25 Branches by Total Loan Amount")

    df_branch = df.groupby('Branch_Name', as_index=False).agg(
        Total_Loan=('Loan_amount', 'sum'),
        Borrower_Count=('BorrowerCode', 'nunique')
    ).sort_values('Total_Loan', ascending=False).head(25)

    # Forest Green color
    chart_branch = alt.Chart(df_branch).mark_bar(color='#1E4D2B').encode(
        x=alt.X('Total_Loan', title='Total Loan Amount'),
        y=alt.Y('Branch_Name', sort='-x', title='Branch Name (Top 25)'),
        tooltip=['Branch_Name', alt.Tooltip('Total_Loan', format=',.0f'), 'Borrower_Count']
    ).properties(
        title='Top 25 Branches by Total Loan Amount',
        height=650
    )
    st.altair_chart(chart_branch, use_container_width=True)


with col_b_right:
    # -------------------------
    # Plot 18: Total Loan Amount by Sc_Rate Bins (Histogram style) 
    # -------------------------
    st.markdown("### Total Loan Amount by Success Sc_Rate Bins")
    
    # Gold color
    chart_bin = alt.Chart(df).mark_bar(color='#FFCC00').encode( 
        x=alt.X('Sc_Rate', bin=alt.Bin(maxbins=10), title='Success Sc_Rate Bin (%)'),
        y=alt.Y('Loan_amount', aggregate='sum', title='Total Loan Amount'),
        tooltip=[
            alt.Tooltip('Sc_Rate', bin=True, title='Sc_Rate Range'),
            alt.Tooltip('Loan_amount', aggregate='sum', format=',.0f', title='Total Loan')
        ]
    ).properties(
        title='Total Loan Amount by Success Rate Bins',
        height=650
    )
    st.altair_chart(chart_bin, use_container_width=True)
    
    # --- 8. PREDICTION ANALYSIS (SARIMAX Implementation - DAILY) ---
@st.cache_data
def run_prediction_analysis(df_in):
    # Ensure data is only for the filtered date range for relevant prediction
    daily_data = df_in.dropna(subset=['Disbursment_Date']).set_index('Disbursment_Date')['Loan_amount'].resample('D').sum()
    daily_data = daily_data.fillna(0)

    if len(daily_data) < 7: # Use a higher threshold for SARIMAX for stability
        last_date = daily_data.index.max()
        next_date = last_date + pd.DateOffset(days=1) if not daily_data.empty and not pd.isna(last_date) else pd.to_datetime(date.today())
        return {
            'last_period_name': daily_data.index.max().strftime('%Y-%m-%d') if not daily_data.empty and not pd.isna(daily_data.index.max()) else 'N/A',
            'next_period_name': next_date.strftime('%Y-%m-%d'),
            'last_period_loan': daily_data.iloc[-1] if not daily_data.empty else 0,
            'next_period_predicted_loan': 0,
            'trend': 'N/A',
            'change_direction': 'N/A',
            'probability_change': 'N/A',
            'plot': None
        }, "Insufficient daily data (less than 7 days) in the selected range for time series analysis."


    # Model parameters: SARIMAX (1, 1, 0)
    order = (1, 1, 0) 
    seasonal_order = (0, 0, 0, 0)

    try:
        best_model = SARIMAX(
            endog=daily_data,
            order=order,
            seasonal_order=seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False
        )
        model_fit = best_model.fit(disp=False)
    except Exception as e:
        return None, f"Model fitting failed with SARIMAX(1, 1, 0): {e}"

    # Predict the next day
    future_steps = 1
    start_date_forecast = daily_data.index[-1] + pd.DateOffset(days=1)
    forecast_index = pd.date_range(start=start_date_forecast, periods=future_steps, freq='D')

    forecast_obj = model_fit.get_prediction(start=forecast_index[0], end=forecast_index[-1])
    forecast_result = forecast_obj.predicted_mean

    # Extract values and format for daily prediction
    last_period_loan = daily_data.iloc[-1]
    next_period_predicted_loan = max(0, forecast_result.iloc[0]) 
    last_period_name = daily_data.index.max().strftime('%Y-%m-%d')
    next_period_name = forecast_index[0].strftime('%Y-%m-%d')
    
    # Analyze the trend 
    if last_period_loan > 0:
        change_ratio = next_period_predicted_loan / last_period_loan
        if change_ratio > 1:
            trend = "INCREASED"
            probability_change = f"{(change_ratio - 1) * 100:.2f}%"
        else:
            trend = "DECREASED"
            probability_change = f"{(1 - change_ratio) * 100:.2f}%"
    else: # Handle division by zero if last_period_loan is 0
        trend = "INCREASED" if next_period_predicted_loan > 0 else "NO CHANGE"
        probability_change = 'N/A'


    # Generate Matplotlib plot for DAILY data
    # --- INCREASED FIGURE SIZE FOR BETTER VISIBILITY ---
    plt.figure(figsize=(13, 4)) 
    plt.plot(daily_data.index, daily_data.values, marker='o', color='#1E4D2B', label='Historical Loan Amount') 
    
    forecast_color = '#008000' if trend == 'INCREASED' else '#CC0000'
    plt.plot(forecast_index, forecast_result.values, color=forecast_color, marker='*', markersize=10, linestyle='--', label=f'Forecasted Loan Amount ({next_period_name})')
    
    plt.title('Daily Loan Disbursement Forecast', color='#1E4D2B')
    plt.xlabel('Date')
    plt.ylabel('Total Loan Amount')
    plt.legend()
    plt.grid(True)
    plot_fig = plt.gcf()
    plt.close(plot_fig)

    results = {
        'last_period_name': last_period_name,
        'next_period_name': next_period_name,
        'last_period_loan': last_period_loan,
        'next_period_predicted_loan': next_period_predicted_loan,
        'trend': trend,
        'change_direction': trend.lower(),
        'probability_change': probability_change,
        'plot': plot_fig
    }
    return results, None

prediction_results, error_message = run_prediction_analysis(df)

st.markdown("---")
st.header("AI Prediction: Next Day's Loan Trend Forecast 🚀")

if error_message:
    st.warning(f"Prediction Warning: {error_message}")
elif prediction_results:
    results = prediction_results
    
    trend_class = 'trend-increased' if results['trend'] == 'INCREASED' else 'trend-decreased'
    
    # Updated text for daily prediction
    st.markdown(f"""
    <div class="prediction-box">
        <h3>Prediction: Loan Disbursement is Expected to <span class="{trend_class}">{results['trend']}</span> on **{results['next_period_name']}**.</h3>
        <p>Model used: SARIMAX(1, 1, 0). Data used: {len(df.dropna(subset=['Disbursment_Date']).set_index('Disbursment_Date')['Loan_amount'].resample('D').sum())} days of historical daily data (filtered by date selection).</p>
        <ul>
            <li>** Last Day ({results['last_period_name']}):** {results['last_period_loan']:,.0f}</li>
            <li>** Forecast ({results['next_period_name']}):** {results['next_period_predicted_loan']:,.0f} (A change of {results['probability_change']})</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Display the plot
    st.pyplot(results['plot'])
