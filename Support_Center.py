# streamlit_app.py
# -------------------------------------------------------------
# Support Center Issue Analysis – Interactive Streamlit Dashboard
# Reads from an uploaded Excel *or* a default path and analyzes
# Operation_Code, Zone_Name, Assign_Name performance.
# -------------------------------------------------------------

import os
import io
import numpy as np
import pandas as pd
import streamlit as st
from datetime import datetime

# Optional interactive charts (fallback to matplotlib if plotly not available)
try:
    import plotly.express as px
    PLOTLY_OK = True
except Exception:
    PLOTLY_OK = False

# -----------------------------
# Config
# -----------------------------
st.set_page_config(page_title="Support Center Performance Dashboard", layout="wide")
st.title("📊 Support Center Issue Analysis")
st.caption("Analyze performance by **Assignee** and **Zone** from your Support Center issue log.")

# -----------------------------
# Helpers
# -----------------------------
REQ_COLS = [
    "Operation_Code",      # e.g., OP-001
    "Zone_Name",           # e.g., Dhaka, Rajshahi
    "Assign_Name",         # e.g., Alice
    "Issue_Date",          # date opened
    "Solve_Date",          # date closed (optional)
    "Days_Taken",          # numeric (optional – will be computed if missing)
    "Status_Description",  # e.g., Solved, Working, On-Hold
]

@st.cache_data(show_spinner=False)
def load_excel(file_bytes_or_path):
    """Load Excel from an uploaded file or a path. Normalize columns and types."""
    if isinstance(file_bytes_or_path, (bytes, bytearray, io.BytesIO)):
        df = pd.read_excel(file_bytes_or_path)
    else:
        df = pd.read_excel(file_bytes_or_path)

    # Normalize column names: strip, lower, replace spaces/underscores uniformly
    rename_map = {}
    for c in df.columns:
        key = str(c).strip().lower().replace(" ", "_")
        rename_map[c] = key
    df = df.rename(columns=rename_map)

    # Build a flexible mapping to expected names
    candidates = {
        "operation_code": ["operation_code", "op_code", "ticket_id", "id", "code"],
        "zone_name": ["zone_name", "zone", "region"],
        "assign_name": ["assign_name", "assignee", "assigned_to", "owner"],
        "issue_date": ["issue_date", "created", "open_date", "opened_on", "date"],
        "solve_date": ["solve_date", "resolved", "close_date", "closed_on", "resolution_date"],
        "days_taken": ["days_taken", "days", "resolution_days", "age_days"],
        "status_description": ["status_description", "status", "state"],
    }

    def first_match(possible):
        for name in possible:
            if name in df.columns:
                return name
        return None

    colmap = {k: first_match(v) for k, v in candidates.items()}

    # Create a normalized DataFrame with expected columns (some may be None)
    nd = pd.DataFrame()
    for target, src in colmap.items():
        if src is not None:
            nd[target] = df[src]
        else:
            nd[target] = np.nan

    # Type conversions
    for dcol in ["issue_date", "solve_date"]:
        nd[dcol] = pd.to_datetime(nd[dcol], errors="coerce")

    # If days_taken missing, compute from dates where possible
    if nd["days_taken"].isna().all():
        nd["days_taken"] = (nd["solve_date"] - nd["issue_date"]).dt.days

    # Clean strings
    for s in ["status_description", "assign_name", "zone_name", "operation_code"]:
        nd[s] = nd[s].astype(str).str.strip()
        nd.loc[nd[s].isin(["nan", "None"]), s] = np.nan

    # Final friendly column names
    nd = nd.rename(
        columns={
            "operation_code": "Operation_Code",
            "zone_name": "Zone_Name",
            "assign_name": "Assign_Name",
            "issue_date": "Issue_Date",
            "solve_date": "Solve_Date",
            "days_taken": "Days_Taken",
            "status_description": "Status_Description",
        }
    )

    # Remove fully empty rows on key fields
    nd = nd.dropna(subset=["Operation_Code", "Issue_Date"], how="all")

    # Ensure numeric Days_Taken
    nd["Days_Taken"] = pd.to_numeric(nd["Days_Taken"], errors="coerce")

    return nd.reset_index(drop=True)


def compute_metrics(df: pd.DataFrame):
    """Compute core metrics for KPIs and leaderboards."""
    df = df.copy()
    # If status missing, infer: solved if Solve_Date present and Days_Taken >= 0
    if "Status_Description" not in df or df["Status_Description"].isna().all():
        df["Status_Description"] = np.where(df["Solve_Date"].notna(), "Solved", "Open")

    solved_mask = df["Status_Description"].str.lower().eq("solved")

    # KPI
    total_issues = len(df)
    solved_issues = int(solved_mask.sum())
    solved_rate = (solved_issues / total_issues) if total_issues else np.nan
    avg_days_overall = df.loc[solved_mask, "Days_Taken"].mean()

    # Per-assignee
    grp_a = df.groupby("Assign_Name", dropna=False)
    per_assignee = grp_a.agg(
        Total_Issues=("Operation_Code", "count"),
        Solved_Issues=("Status_Description", lambda s: (s.str.lower()=="solved").sum()),
        Avg_Days_Resolved=("Days_Taken", lambda s: s[df.loc[s.index, "Status_Description"].str.lower()=="solved"].mean()),
        Median_Days_Resolved=("Days_Taken", lambda s: s[df.loc[s.index, "Status_Description"].str.lower()=="solved"].median()),
    )
    per_assignee["Solve_Rate"] = per_assignee["Solved_Issues"] / per_assignee["Total_Issues"].replace(0, np.nan)

    # Simple performance score: lower days + higher solve rate
    # Normalize using z-scores (robust to NaN)
    z_sr = (per_assignee["Solve_Rate"] - per_assignee["Solve_Rate"].mean()) / per_assignee["Solve_Rate"].std(ddof=0)
    z_days = (per_assignee["Avg_Days_Resolved"] - per_assignee["Avg_Days_Resolved"].mean()) / per_assignee["Avg_Days_Resolved"].std(ddof=0)
    per_assignee["Performance_Score"] = (z_sr.fillna(0) - z_days.fillna(0))

    # Per-zone
    grp_z = df.groupby("Zone_Name", dropna=False)
    per_zone = grp_z.agg(
        Total_Issues=("Operation_Code", "count"),
        Solved_Issues=("Status_Description", lambda s: (s.str.lower()=="solved").sum()),
        Avg_Days_Resolved=("Days_Taken", lambda s: s[df.loc[s.index, "Status_Description"].str.lower()=="solved"].mean()),
    )
    per_zone["Solve_Rate"] = per_zone["Solved_Issues"] / per_zone["Total_Issues"].replace(0, np.nan)

    return {
        "kpi": {
            "total": total_issues,
            "solved": solved_issues,
            "solve_rate": solved_rate,
            "avg_days": avg_days_overall,
        },
        "per_assignee": per_assignee.sort_values(["Performance_Score"], ascending=False),
        "per_zone": per_zone.sort_values(["Solve_Rate", "Avg_Days_Resolved"], ascending=[False, True]),
    }

# -----------------------------
# Data Ingest UI
# -----------------------------
DEFAULT_PATH = "/mnt/data/Issue List of Support Center.xlsx"  # Provided upload path

left, right = st.columns([1,2])
with left:
    uploaded = st.file_uploader("Upload Excel (xlsx)", type=["xlsx"], help="Upload the Support Center issue log.")

    use_default = False
    if uploaded is None:
        if os.path.exists(DEFAULT_PATH):
            use_default = st.toggle("Use default uploaded file", value=True, help=f"{DEFAULT_PATH}")
        else:
            st.info("No file uploaded yet. Please upload an .xlsx file.")

# Load data
if uploaded is not None:
    df_all = load_excel(uploaded)
elif 'use_default' in locals() and use_default:
    df_all = load_excel(DEFAULT_PATH)
else:
    df_all = pd.DataFrame(columns=["Operation_Code","Zone_Name","Assign_Name","Issue_Date","Solve_Date","Days_Taken","Status_Description"])  # empty placeholder

# Guard
if df_all.empty:
    st.warning("No data loaded. Upload a file or enable the default path.")
    st.stop()

# -----------------------------
# Filters
# -----------------------------
st.sidebar.header("🔎 Filters")

min_date = pd.to_datetime(df_all["Issue_Date"].min()) if df_all["Issue_Date"].notna().any() else None
max_date = pd.to_datetime(df_all["Issue_Date"].max()) if df_all["Issue_Date"].notna().any() else None

if min_date and max_date:
    date_range = st.sidebar.date_input(
        "Issue Date Range",
        value=(min_date.date(), max_date.date()),
        min_value=min_date.date(),
        max_value=max_date.date(),
    )
else:
    date_range = None

zones = sorted([z for z in df_all["Zone_Name"].dropna().unique()])
assignees = sorted([a for a in df_all["Assign_Name"].dropna().unique()])
statuses = sorted([s for s in df_all["Status_Description"].dropna().unique()])

sel_zones = st.sidebar.multiselect("Zones", options=zones, default=zones)
sel_assignees = st.sidebar.multiselect("Assignees", options=assignees, default=assignees)
sel_status = st.sidebar.multiselect("Status", options=statuses, default=statuses if statuses else [])

# Apply filters
fdf = df_all.copy()
if date_range:
    start, end = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
    fdf = fdf[(fdf["Issue_Date"] >= start) & (fdf["Issue_Date"] <= end)]
if sel_zones:
    fdf = fdf[fdf["Zone_Name"].isin(sel_zones)]
if sel_assignees:
    fdf = fdf[fdf["Assign_Name"].isin(sel_assignees)]
if sel_status:
    fdf = fdf[fdf["Status_Description"].isin(sel_status)]

# -----------------------------
# KPIs
# -----------------------------
metrics = compute_metrics(fdf)
mk = metrics["kpi"]

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Issues", f"{mk['total']:,}")
c2.metric("Solved Issues", f"{mk['solved']:,}")
c3.metric("Solve Rate", f"{mk['solve_rate']*100:,.1f}%" if not pd.isna(mk['solve_rate']) else "N/A")
c4.metric("Avg Days to Resolve", f"{mk['avg_days']:.1f}" if not pd.isna(mk['avg_days']) else "N/A")

st.divider()

# -----------------------------
# Leaderboards and Visuals
# -----------------------------
colA, colB = st.columns(2)

with colA:
    st.subheader("🏅 Assignee Leaderboard (Who did better)")
    pa = metrics["per_assignee"].copy()
    if not pa.empty:
        pa_display = pa[["Total_Issues","Solved_Issues","Solve_Rate","Avg_Days_Resolved","Median_Days_Resolved","Performance_Score"]].round(2)
        st.dataframe(pa_display, use_container_width=True)

        if PLOTLY_OK:
            fig = px.bar(
                pa.reset_index().rename(columns={"index":"Assign_Name"}),
                x="Assign_Name", y="Performance_Score",
                hover_data=["Solve_Rate","Avg_Days_Resolved","Total_Issues"],
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Plotly not available, showing table only.")
    else:
        st.info("No assignee data to display.")

with colB:
    st.subheader("🗺️ Zone Performance (Operation_Code × Zone_Name)")
    pz = metrics["per_zone"].copy()
    if not pz.empty:
        st.dataframe(pz[["Total_Issues","Solved_Issues","Solve_Rate","Avg_Days_Resolved"]].round(2), use_container_width=True)
        if PLOTLY_OK:
            fig2 = px.scatter(
                pz.reset_index().rename(columns={"index":"Zone_Name"}),
                x="Avg_Days_Resolved", y="Solve_Rate", size="Total_Issues", color="Zone_Name",
                hover_name="Zone_Name",
            )
            fig2.update_layout(yaxis_tickformat=",.0%")
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("Plotly not available, showing table only.")
    else:
        st.info("No zone data to display.")

st.divider()

# -----------------------------
# Trends
# -----------------------------
st.subheader("📈 Monthly Trend by Status")
if not fdf.empty and fdf["Issue_Date"].notna().any():
    tmp = fdf.copy()
    tmp["Month"] = tmp["Issue_Date"].dt.to_period("M").astype(str)
    trend = tmp.groupby(["Month", "Status_Description"]).size().reset_index(name="Count")
    trend = trend.sort_values("Month")

    if not trend.empty and PLOTLY_OK:
        fig3 = px.line(trend, x="Month", y="Count", color="Status_Description", markers=True)
        st.plotly_chart(fig3, use_container_width=True)
    else:
        st.dataframe(trend, use_container_width=True)
else:
    st.info("No trend data available.")

st.divider()

# -----------------------------
# Detailed Table + Download
# -----------------------------
st.subheader("🔎 Issue Details (filtered)")
show_cols = [c for c in ["Operation_Code","Zone_Name","Assign_Name","Issue_Date","Solve_Date","Days_Taken","Status_Description"] if c in fdf.columns]
st.dataframe(
    fdf.sort_values(by=["Issue_Date"], ascending=False)[show_cols],
    height=350,
    use_container_width=True,
)

csv = fdf[show_cols].to_csv(index=False).encode("utf-8")
st.download_button("⬇️ Download filtered CSV", data=csv, file_name="support_center_filtered.csv", mime="text/csv")

# -----------------------------
# Insights – Who can do better in the future
# -----------------------------
st.subheader("💡 Insights & Opportunities (Who can do better)")
if not metrics["per_assignee"].empty:
    pa = metrics["per_assignee"].copy()
    # Candidates who handled >= median issues but below-median solve rate or above-median avg days
    vol_threshold = pa["Total_Issues"].median(skipna=True)
    sr_median = pa["Solve_Rate"].median(skipna=True)
    days_median = pa["Avg_Days_Resolved"].median(skipna=True)

    can_improve = pa[(pa["Total_Issues"] >= vol_threshold) & ((pa["Solve_Rate"] < sr_median) | (pa["Avg_Days_Resolved"] > days_median))]
    if not can_improve.empty:
        st.write("**Focus coaching on:**")
        st.dataframe(can_improve[["Total_Issues","Solve_Rate","Avg_Days_Resolved","Median_Days_Resolved","Performance_Score"]].round(2), use_container_width=True)
    else:
        st.success("Great! No clear improvement candidates based on current thresholds.")

    # Quick tips
    st.markdown(
        """
        - Prioritize aging tickets in zones with **higher avg days**.
        - Balance workload: redistribute from assignees with **long Avg Days** and **low Solve Rate**.
        - Set an SLA target (e.g., *≤ 5 days*) and track breach counts by assignee/zone.
        """
    )
else:
    st.info("Insufficient data to derive improvement insights.")

st.caption("Built with Streamlit. Drop in your Excel and explore performance by Operation Code, Zone, and Assignee.")
