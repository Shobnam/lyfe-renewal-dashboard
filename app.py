import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="Lyfe Renewal Possibility", page_icon="🩺", layout="wide")
st.title("🩺 Lyfe Renewal Possibility App")

# ---------------- SIDEBAR ----------------
st.sidebar.header("⚖️ Renewal Score Weightage")
w_lcm = st.sidebar.slider("LCM Engagement", 0, 30, 10)
w_hc = st.sidebar.slider("HC Engagement", 0, 30, 10)
w_ct = st.sidebar.slider("CT Engagement", 0, 30, 20)
w_psy = st.sidebar.slider("Psychotherapist Engagement", 0, 30, 15)
w_device = st.sidebar.slider("Device + Vital Usage", 0, 30, 15)
w_clinical = st.sidebar.slider("Clinical Improvement", 0, 30, 20)
w_nps = st.sidebar.slider("NPS", 0, 30, 10)

st.sidebar.divider()
st.sidebar.header("🚨 Risk Penalties")
pen_silent = st.sidebar.slider("Device available but no vitals", 0, 30, 15)
pen_31_60 = st.sidebar.slider("Call gap 31–60 days", 0, 10, 2)
pen_61_90 = st.sidebar.slider("Call gap 61–90 days", 0, 15, 5)
pen_90 = st.sidebar.slider("Call gap >90 days", 0, 20, 8)
pen_no_call = st.sidebar.slider("No completed call", 0, 20, 10)

uploaded_file = st.file_uploader("📂 Upload Patient File (CSV or Excel)", type=["csv", "xlsx"])
if uploaded_file is None:
    st.info("CSV ya Excel file upload karein.")
    st.stop()

# ---------------- READ FILE ----------------
try:
    if uploaded_file.name.lower().endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        xls = pd.ExcelFile(uploaded_file, engine="openpyxl")
        detected = None
        for s in xls.sheet_names:
            sample = pd.read_excel(uploaded_file, sheet_name=s, engine="openpyxl", nrows=5)
            sample.columns = sample.columns.astype(str).str.strip()
            if "Subscription End Date" in sample.columns:
                detected = s
                break
        selected_sheet = st.selectbox("Select Excel Sheet", xls.sheet_names, index=xls.sheet_names.index(detected) if detected else 0)
        df = pd.read_excel(uploaded_file, sheet_name=selected_sheet, engine="openpyxl")
except Exception as e:
    st.error("File read nahi ho paayi.")
    st.exception(e)
    st.stop()

df.columns = df.columns.astype(str).str.strip()

# Remove exact duplicate column names safely, keep first occurrence
if df.columns.duplicated().any():
    duplicate_names = df.columns[df.columns.duplicated()].tolist()
    st.warning(f"Duplicate columns mile aur first copy retain ki gayi: {duplicate_names}")
    df = df.loc[:, ~df.columns.duplicated()].copy()

# Prefer Patient Name; use Patient Name2 only as fallback
if "Patient Name" not in df.columns and "Patient Name2" in df.columns:
    df = df.rename(columns={"Patient Name2": "Patient Name"})

if "Subscription End Date" not in df.columns:
    st.error("`Subscription End Date` column nahi mila.")
    st.write(df.columns.tolist())
    st.stop()

# ---------------- DATES ----------------
date_cols = [
    "Subscription Start Date", "Subscription End Date", "Last Completed Call Date",
    "Last Vital Date", "Initial Active Weight Date", "Latest Active Weight Date",
    "Baseline Record Date", "Current SBP Date", "Current LDL Date",
    "Current HbA1c Date", "Latest NPS Date"
]
for c in date_cols:
    if c in df.columns:
        df[c] = pd.to_datetime(df[c], errors="coerce")

df = df[df["Subscription End Date"].notna()].copy()
df["Renewal Month"] = df["Subscription End Date"].dt.strftime("%b %Y")
df["Renewal Month Sort"] = df["Subscription End Date"].dt.to_period("M").dt.to_timestamp()

month_table = df[["Renewal Month", "Renewal Month Sort"]].drop_duplicates().sort_values("Renewal Month Sort")
months = month_table["Renewal Month"].tolist()
if not months:
    st.warning("Valid renewal month nahi mila.")
    st.stop()

selected_month = st.selectbox("📅 Select Renewal Month", months)
filtered_df = df[df["Renewal Month"] == selected_month].copy().reset_index(drop=True)

# ---------------- HELPERS ----------------
def num(row, col, default=np.nan):
    if col not in row.index:
        return default
    v = pd.to_numeric(row[col], errors="coerce")
    return default if pd.isna(v) else float(v)

def txt(row, col):
    if col not in row.index or pd.isna(row[col]):
        return ""
    return str(row[col]).strip().upper()

def yes(row, col):
    return txt(row, col) in {"YES", "Y", "TRUE", "1"}

def service_score(row, scheduled_col, completed_col, pct_col, target):
    scheduled = num(row, scheduled_col, 0)
    completed = num(row, completed_col, 0)
    if scheduled <= 0:
        return np.nan
    pct = num(row, pct_col, np.nan)
    if pd.isna(pct):
        pct = completed / scheduled * 100
    pct = np.clip(pct, 0, 100)
    effort = min(completed / target, 1) * 100
    return round(pct * 0.70 + effort * 0.30, 2)

def device_score(row):
    expected = num(row, "Vital Devices Available", np.nan)
    if pd.isna(expected):
        expected = sum([
            yes(row, "BP Monitor Available"), yes(row, "Weight Scale Available"),
            yes(row, "Glucometer Available"), yes(row, "Oximeter Available")
        ])
    if expected <= 0:
        expected = num(row, "Active Device Count", 0)
    if expected <= 0:
        return np.nan

    coverage = num(row, "Device Vital Coverage %", np.nan)
    if pd.isna(coverage):
        used = num(row, "Devices With Recent Vitals", np.nan)
        if pd.isna(used):
            used = 0
            used += int(yes(row, "BP Monitor Available") and num(row, "BP Readings - Last 30 Days", 0) > 0)
            used += int(yes(row, "Weight Scale Available") and num(row, "Weight Readings - Last 30 Days", 0) > 0)
            used += int(yes(row, "Glucometer Available") and num(row, "Glucose Readings - Last 30 Days", 0) > 0)
            used += int(yes(row, "Oximeter Available") and num(row, "SpO2 Readings - Last 30 Days", 0) > 0)
        coverage = used / expected * 100
    coverage = np.clip(coverage, 0, 100)
    total_vitals = num(row, "Total Relevant Vitals - Last 30 Days", 0)
    activity = min(total_vitals / max(expected * 4, 1), 1) * 100
    return round(coverage * 0.80 + activity * 0.20, 2)

def clinical_score(row):
    scores = []
    for col in ["Weight Trend", "SBP Trend", "LDL Trend", "HbA1c Trend"]:
        t = txt(row, col)
        if not t or "NO BASELINE" in t or "NO CURRENT" in t or "NO ACTIVE" in t or "ONLY ONE" in t:
            continue
        if "REDUCED" in t:
            scores.append(100)
        elif "NO CHANGE" in t:
            scores.append(50)
        elif "INCREASED" in t:
            scores.append(0)
    return np.nan if not scores else round(float(np.mean(scores)), 2)

def nps_score(row):
    v = num(row, "Latest NPS Score", np.nan)
    return np.nan if pd.isna(v) else round(float(np.clip(v, 0, 10) * 10), 2)

def is_silent_device(row):
    available = num(row, "Vital Devices Available", num(row, "Active Device Count", 0))
    vitals = num(row, "Total Relevant Vitals - Last 30 Days", 0)
    return available > 0 and vitals == 0

def call_details(row):
    # Use source Days Since column when present; otherwise calculate from date.
    days = num(row, "Days Since Last Completed Call", np.nan)
    if pd.isna(days):
        last_date = pd.to_datetime(row.get("Last Completed Call Date", pd.NaT), errors="coerce")
        if pd.isna(last_date):
            return np.nan, pen_no_call, "🚨 No Completed Call"
        days = max((pd.Timestamp.today().normalize() - last_date.normalize()).days, 0)
    if days <= 30:
        return days, 0, "🟢 Active: 0–30 Days"
    if days <= 60:
        return days, pen_31_60, "🟡 Gap: 31–60 Days"
    if days <= 90:
        return days, pen_61_90, "🟠 Gap: 61–90 Days"
    return days, pen_90, "🔴 Gap: >90 Days"

# ---------------- SCORING ----------------
def score_patient(row):
    values = {
        "LCM": service_score(row, "LCM Calls Scheduled", "LCM Calls Completed", "LCM Completion %", 6),
        "HC": service_score(row, "HC Calls Scheduled", "HC Calls Completed", "HC Completion %", 6),
        "CT": service_score(row, "CT Calls Scheduled", "CT Calls Completed", "CT Completion %", 6),
        "Psychotherapist": service_score(row, "Psychotherapist Sessions Scheduled", "Psychotherapist Sessions Completed", "Psychotherapist Completion %", 4),
        "Device": device_score(row),
        "Clinical": clinical_score(row),
        "NPS": nps_score(row),
    }
    weights = {"LCM": w_lcm, "HC": w_hc, "CT": w_ct, "Psychotherapist": w_psy, "Device": w_device, "Clinical": w_clinical, "NPS": w_nps}
    points = 0.0
    applicable = 0.0
    for k, v in values.items():
        if not pd.isna(v) and weights[k] > 0:
            points += (v / 100) * weights[k]
            applicable += weights[k]
    base = points / applicable * 100 if applicable else 0
    silent = is_silent_device(row)
    device_penalty = pen_silent if silent else 0
    days, call_penalty, call_status = call_details(row)
    final = float(np.clip(base - device_penalty - call_penalty, 0, 100))
    possibility = "🟢 High" if final >= 75 else "🟡 Moderate" if final >= 50 else "🔴 Low"
    return pd.Series({
        "LCM Score": values["LCM"], "HC Score": values["HC"], "CT Score": values["CT"],
        "Psychotherapist Score": values["Psychotherapist"], "Device Vital Score": values["Device"],
        "Clinical Improvement Score": values["Clinical"], "NPS Score Component": values["NPS"],
        "Applicable Weight": applicable, "Base Renewal Score": round(base, 1),
        "Silent Device Flag": "🚨 Device Active But No Recent Vitals" if silent else "OK",
        "Device Penalty": device_penalty,
        # IMPORTANT: do not recreate Days Since Last Completed Call; source column already exists.
        "Call Gap Status": call_status, "Call Gap Penalty": call_penalty,
        "Total Penalty": device_penalty + call_penalty,
        "Renewal Score": round(final, 1), "Renewal Possibility": possibility,
    })

scores = filtered_df.apply(score_patient, axis=1)
filtered_df = pd.concat([filtered_df.reset_index(drop=True), scores.reset_index(drop=True)], axis=1)

# Ensure only one Days Since column and numeric type
if "Days Since Last Completed Call" in filtered_df.columns:
    filtered_df["Days Since Last Completed Call"] = pd.to_numeric(filtered_df["Days Since Last Completed Call"], errors="coerce")
else:
    filtered_df["Days Since Last Completed Call"] = np.nan

# ---------------- SUMMARY ----------------
high_count = int((filtered_df["Renewal Possibility"] == "🟢 High").sum())
moderate_count = int((filtered_df["Renewal Possibility"] == "🟡 Moderate").sum())
low_count = int((filtered_df["Renewal Possibility"] == "🔴 Low").sum())
silent_count = int((filtered_df["Silent Device Flag"] != "OK").sum())
call_gap_count = int((filtered_df["Days Since Last Completed Call"] > 60).fillna(False).sum())

st.subheader(f"📊 {selected_month} Renewal Summary")
c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("Total Patients", len(filtered_df))
c2.metric("🟢 High", high_count)
c3.metric("🟡 Moderate", moderate_count)
c4.metric("🔴 Low", low_count)
c5.metric("🚨 Silent Devices", silent_count)
c6.metric("📞 Call Gap >60d", call_gap_count)

# ---------------- FILTERS ----------------
st.divider()
f1, f2, f3 = st.columns(3)
selected_possibility = f1.multiselect("Renewal Possibility", ["🟢 High", "🟡 Moderate", "🔴 Low"], default=["🟢 High", "🟡 Moderate", "🔴 Low"])
only_silent = f2.checkbox("Only Silent Device Patients")
only_gap = f3.checkbox("Only Call Gap >60 Days")

view_df = filtered_df[filtered_df["Renewal Possibility"].isin(selected_possibility)].copy()
if only_silent:
    view_df = view_df[view_df["Silent Device Flag"] != "OK"].copy()
if only_gap:
    view_df = view_df[view_df["Days Since Last Completed Call"] > 60].copy()

# ---------------- TABLE ----------------
st.subheader("🎯 Renewal Possibility")
show_cols = [
    "Patient ID", "Identifier", "Patient Name", "Subscription End Date", "Latest Engagement Status",
    "HC Name", "LCM Name", "CT Physio Name", "Last Completed Call Date", "Days Since Last Completed Call",
    "LCM Calls Scheduled", "LCM Calls Completed", "LCM Completion %",
    "HC Calls Scheduled", "HC Calls Completed", "HC Completion %",
    "CT Calls Scheduled", "CT Calls Completed", "CT Completion %",
    "Psychotherapist Sessions Scheduled", "Psychotherapist Sessions Completed", "Psychotherapist Completion %",
    "Active Device Count", "Vital Devices Available", "Devices With Recent Vitals", "Device Vital Coverage %",
    "Total Relevant Vitals - Last 30 Days", "Silent Device Flag",
    "Weight Trend", "SBP Trend", "LDL Trend", "HbA1c Trend", "Latest NPS Score",
    "Call Gap Status", "Base Renewal Score", "Total Penalty", "Renewal Score", "Renewal Possibility"
]
show_cols = [c for c in show_cols if c in view_df.columns]
st.dataframe(view_df[show_cols].sort_values("Renewal Score", ascending=False), use_container_width=True, hide_index=True)

# ---------------- DOWNLOAD ----------------
st.download_button(
    f"📥 Download {selected_month} Scored Patients",
    view_df.drop(columns=["Renewal Month Sort"], errors="ignore").to_csv(index=False).encode("utf-8"),
    f"{selected_month.replace(' ', '_').lower()}_renewal_possibility.csv",
    "text/csv"
)

st.info("Business-rule score hai; actual renewal probability nahi. Historical renewed/non-renewed data se weights validate karna chahiye.")
