import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="Lyfe Renewal Intelligence", page_icon="🩺", layout="wide")
st.title("🩺 Lyfe Renewal Intelligence")
st.caption("Manual weightage score aur 2026 ke actually renewed patients ke pattern se similarity, dono alag columns mein.")

CURRENT_SHEET = "Current_Renewal_Patients"
HISTORY_SHEET = "Historical_Renewed_2026"

# ---------------- Sidebar ----------------
st.sidebar.header("⚖️ Manual Weightage")
manual_weights = {
    "LCM": st.sidebar.slider("LCM", 0, 30, 10),
    "HC": st.sidebar.slider("HC", 0, 30, 10),
    "CT": st.sidebar.slider("CT", 0, 30, 20),
    "Psychotherapist": st.sidebar.slider("Psychotherapist", 0, 30, 15),
    "Device/Vitals": st.sidebar.slider("Device + Vitals", 0, 30, 15),
    "Clinical": st.sidebar.slider("Clinical Improvement", 0, 30, 20),
    "NPS": st.sidebar.slider("NPS", 0, 30, 10),
}
st.sidebar.metric("Total Manual Weight", sum(manual_weights.values()))

st.sidebar.divider()
st.sidebar.header("🚨 Penalties")
pen_silent = st.sidebar.slider("Device but no recent vitals", 0, 30, 15)
pen_31_60 = st.sidebar.slider("Call gap 31–60 days", 0, 10, 2)
pen_61_90 = st.sidebar.slider("Call gap 61–90 days", 0, 15, 5)
pen_90 = st.sidebar.slider("Call gap >90 days", 0, 20, 8)
pen_no_call = st.sidebar.slider("No completed call", 0, 20, 10)

file = st.file_uploader("📂 Upload patients_data_2026 Excel", type=["xlsx"])
if file is None:
    st.info(f"Excel upload karein. Required tabs: `{CURRENT_SHEET}` and `{HISTORY_SHEET}`")
    st.stop()

# ---------------- Read workbook ----------------
try:
    xls = pd.ExcelFile(file, engine="openpyxl")
    missing = [s for s in [CURRENT_SHEET, HISTORY_SHEET] if s not in xls.sheet_names]
    if missing:
        st.error(f"Missing Excel tabs: {missing}")
        st.write("Available tabs:", xls.sheet_names)
        st.stop()
    current = pd.read_excel(file, sheet_name=CURRENT_SHEET, engine="openpyxl")
    history = pd.read_excel(file, sheet_name=HISTORY_SHEET, engine="openpyxl")
except Exception as e:
    st.error("Excel read nahi hua.")
    st.exception(e)
    st.stop()


def clean_df(df):
    df = df.copy()
    df.columns = df.columns.astype(str).str.strip()
    df = df.loc[:, ~df.columns.duplicated()].copy()
    if "Patient Name" not in df.columns and "Patient Name2" in df.columns:
        df = df.rename(columns={"Patient Name2": "Patient Name"})
    return df

current = clean_df(current)
history = clean_df(history)

# ---------------- Helpers ----------------
def nseries(df, col, default=np.nan):
    if col not in df.columns:
        return pd.Series(default, index=df.index, dtype=float)
    return pd.to_numeric(df[col], errors="coerce")


def first_numeric(df, names, default=np.nan):
    out = pd.Series(np.nan, index=df.index, dtype=float)
    for name in names:
        if name in df.columns:
            out = out.fillna(pd.to_numeric(df[name], errors="coerce"))
    return out.fillna(default) if not pd.isna(default) else out


def service_component(df, scheduled, completed, pct, target):
    sch = nseries(df, scheduled, 0).fillna(0)
    comp = nseries(df, completed, 0).fillna(0)
    rate = nseries(df, pct)
    rate = rate.fillna(np.where(sch > 0, comp / sch * 100, np.nan))
    effort = (comp / target).clip(0, 1) * 100
    score = rate.clip(0, 100) * 0.70 + effort * 0.30
    return score.where(sch > 0)


def yes_series(df, col):
    if col not in df.columns:
        return pd.Series(False, index=df.index)
    return df[col].astype(str).str.strip().str.upper().isin(["YES", "Y", "TRUE", "1"])


def build_components(df, historical=False):
    c = pd.DataFrame(index=df.index)
    c["LCM"] = service_component(df, "LCM Calls Scheduled", "LCM Calls Completed", "LCM Completion %", 6)
    c["HC"] = service_component(df, "HC Calls Scheduled", "HC Calls Completed", "HC Completion %", 6)
    c["CT"] = service_component(df, "CT Calls Scheduled", "CT Calls Completed", "CT Completion %", 6)
    c["Psychotherapist"] = service_component(
        df, "Psychotherapist Sessions Scheduled", "Psychotherapist Sessions Completed",
        "Psychotherapist Completion %", 4
    )

    available = first_numeric(df, ["Vital Devices Available", "Active Device Count"])
    if available.isna().all():
        available = (
            yes_series(df, "BP Monitor Available").astype(int)
            + yes_series(df, "Weight Scale Available").astype(int)
            + yes_series(df, "Glucometer Available").astype(int)
            + yes_series(df, "Oximeter Available").astype(int)
        ).astype(float)

    recent_vitals = first_numeric(
        df,
        ["Total Relevant Vitals - Last 30 Days", "Vital Readings - Last 30 Days of Plan"],
        0,
    ).fillna(0)
    coverage = first_numeric(df, ["Device Vital Coverage %"])
    devices_used = first_numeric(df, ["Devices With Recent Vitals"])
    inferred_coverage = np.where(available > 0, devices_used / available * 100, np.nan)
    coverage = coverage.fillna(pd.Series(inferred_coverage, index=df.index))
    activity = (recent_vitals / (available.clip(lower=1) * 4)).clip(0, 1) * 100
    c["Device/Vitals"] = (coverage.clip(0, 100) * 0.80 + activity * 0.20).where(available > 0)

    trend_cols = ["Weight Trend", "SBP Trend", "LDL Trend", "HbA1c Trend"]
    trend_scores = []
    for col in trend_cols:
        if col in df.columns:
            t = df[col].astype(str).str.upper()
            s = pd.Series(np.nan, index=df.index)
            s[t.str.contains("REDUCED", na=False)] = 100
            s[t.str.contains("NO CHANGE", na=False)] = 50
            s[t.str.contains("INCREASED", na=False)] = 0
            trend_scores.append(s)

    # Historical query has no SBP/LDL/HbA1c trends. Use available outcome values as neutral evidence,
    # while weight trend remains outcome-based.
    if historical:
        for col in ["Latest SBP - Plan", "Latest LDL - Plan", "Latest HbA1c - Plan"]:
            if col in df.columns:
                v = pd.to_numeric(df[col], errors="coerce")
                trend_scores.append(pd.Series(np.where(v.notna(), 50, np.nan), index=df.index))

    c["Clinical"] = pd.concat(trend_scores, axis=1).mean(axis=1, skipna=True) if trend_scores else np.nan
    c["NPS"] = nseries(df, "Latest NPS Score").clip(0, 10) * 10
    return c, available, recent_vitals


def weighted_score(components, weights):
    numerator = pd.Series(0.0, index=components.index)
    denominator = pd.Series(0.0, index=components.index)
    for col, weight in weights.items():
        valid = components[col].notna()
        numerator += components[col].fillna(0) / 100 * weight
        denominator += valid.astype(float) * weight
    return (numerator / denominator.replace(0, np.nan) * 100).fillna(0)


def band(score, high=75, moderate=50):
    return np.select(
        [score >= high, score >= moderate],
        ["🟢 High", "🟡 Moderate"],
        default="🔴 Low",
    )

# ---------------- Current month ----------------
for col in ["Subscription End Date", "Last Completed Call Date"]:
    if col in current.columns:
        current[col] = pd.to_datetime(current[col], errors="coerce")

if "Subscription End Date" not in current.columns:
    st.error("Current sheet mein `Subscription End Date` missing hai.")
    st.stop()

current = current[current["Subscription End Date"].notna()].copy()
current["Renewal Month"] = current["Subscription End Date"].dt.strftime("%b %Y")
current["Renewal Month Sort"] = current["Subscription End Date"].dt.to_period("M").dt.to_timestamp()
month_list = (
    current[["Renewal Month", "Renewal Month Sort"]]
    .drop_duplicates().sort_values("Renewal Month Sort")["Renewal Month"].tolist()
)
selected_month = st.selectbox("📅 Select Renewal Month", month_list)
current_month = current[current["Renewal Month"] == selected_month].copy().reset_index(drop=True)

# ---------------- Components and history profile ----------------
current_comp, current_devices, current_recent_vitals = build_components(current_month, historical=False)
history_comp, _, _ = build_components(history, historical=True)

history_profile = history_comp.mean(skipna=True)
history_valid_count = history_comp.notna().sum()

# Historical weights: stable renewed-pattern strength, not predictive coefficients.
# A factor receives more weight when renewed patients consistently have a high component score.
history_raw = (history_profile.fillna(0) / 100) * np.sqrt(history_valid_count.clip(lower=0))
if history_raw.sum() > 0:
    history_weights = history_raw / history_raw.sum() * 100
else:
    history_weights = pd.Series(100 / len(history_profile), index=history_profile.index)

# ---------------- Manual score ----------------
manual_base = weighted_score(current_comp, manual_weights)

source_gap = first_numeric(current_month, ["Days Since Last Completed Call"])
if source_gap.isna().all() and "Last Completed Call Date" in current_month.columns:
    source_gap = (pd.Timestamp.today().normalize() - current_month["Last Completed Call Date"].dt.normalize()).dt.days

call_penalty = pd.Series(0.0, index=current_month.index)
call_penalty[source_gap.between(31, 60, inclusive="both")] = pen_31_60
call_penalty[source_gap.between(61, 90, inclusive="both")] = pen_61_90
call_penalty[source_gap > 90] = pen_90
call_penalty[source_gap.isna()] = pen_no_call

silent_flag = (current_devices > 0) & (current_recent_vitals == 0)
device_penalty = silent_flag.astype(float) * pen_silent
manual_final = (manual_base - call_penalty - device_penalty).clip(0, 100)

current_month["Manual Renewal Score"] = manual_final.round(1)
current_month["Manual Possibility"] = band(manual_final)
current_month["Call Gap Penalty"] = call_penalty
current_month["Device Penalty"] = device_penalty
current_month["Silent Device Flag"] = np.where(silent_flag, "🚨 Device Active But No Recent Vitals", "OK")

# ---------------- Historical renewed pattern similarity ----------------
# Similarity per component: 100 at renewed mean; gradually decreases with distance.
similarity = pd.DataFrame(index=current_comp.index)
for col in current_comp.columns:
    target = history_profile.get(col, np.nan)
    if pd.isna(target):
        similarity[col] = np.nan
    else:
        similarity[col] = (100 - (current_comp[col] - target).abs()).clip(0, 100)

history_weight_dict = history_weights.to_dict()
historical_match_score = weighted_score(similarity, history_weight_dict)
current_month["Historical Renewed Pattern Score"] = historical_match_score.round(1)
current_month["Historical Pattern Match"] = band(historical_match_score, high=80, moderate=60)

# ---------------- Tabs ----------------
tab1, tab2, tab3 = st.tabs([
    "🎯 Current Renewal Possibility",
    "📚 Historical Renewed Pattern",
    "⚖️ Weightage Comparison",
])

with tab1:
    st.subheader(f"{selected_month} Current Renewal Patients")
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("Total", len(current_month))
    m2.metric("Manual High", int((current_month["Manual Possibility"] == "🟢 High").sum()))
    m3.metric("Manual Moderate", int((current_month["Manual Possibility"] == "🟡 Moderate").sum()))
    m4.metric("Manual Low", int((current_month["Manual Possibility"] == "🔴 Low").sum()))
    m5.metric("Historical Strong Match", int((current_month["Historical Pattern Match"] == "🟢 High").sum()))
    m6.metric("Silent Devices", int(silent_flag.sum()))

    show_cols = [
        "Patient ID", "Identifier", "Patient Name", "Subscription End Date",
        "LCM Calls Completed", "HC Calls Completed", "CT Calls Completed",
        "Psychotherapist Sessions Completed", "Latest NPS Score",
        "Last Completed Call Date", "Days Since Last Completed Call",
        "Silent Device Flag", "Manual Renewal Score", "Manual Possibility",
        "Historical Renewed Pattern Score", "Historical Pattern Match",
    ]
    show_cols = [c for c in show_cols if c in current_month.columns]
    st.dataframe(
        current_month[show_cols].sort_values(
            ["Historical Renewed Pattern Score", "Manual Renewal Score"], ascending=False
        ),
        use_container_width=True,
        hide_index=True,
    )

    st.download_button(
        "📥 Download scored current patients",
        current_month.drop(columns=["Renewal Month Sort"], errors="ignore").to_csv(index=False).encode("utf-8"),
        f"{selected_month.replace(' ', '_').lower()}_manual_historical_scores.csv",
        "text/csv",
    )

with tab2:
    st.subheader("2026 Actually Renewed Patients Pattern")
    st.metric("Historical renewed rows", len(history))

    profile_df = pd.DataFrame({
        "Factor": history_profile.index,
        "Renewed Patient Average Score": history_profile.values,
        "Available Historical Records": history_valid_count.reindex(history_profile.index).values,
        "Historical Renewed Weightage %": history_weights.reindex(history_profile.index).values,
    }).round(2)
    st.dataframe(profile_df, use_container_width=True, hide_index=True)

    hist_cols = [
        "Identifier", "Patient Name", "Plan Name", "Plan Start Date", "Plan End Date",
        "Next Plan Name", "Next Plan Start Date", "Days to Next Plan",
        "LCM Completion %", "HC Completion %", "CT Completion %",
        "Psychotherapist Completion %", "Vital Readings - Last 30 Days of Plan",
        "Weight Change %", "Latest NPS Score",
    ]
    hist_cols = [c for c in hist_cols if c in history.columns]
    st.dataframe(history[hist_cols], use_container_width=True, hide_index=True)

with tab3:
    st.subheader("Manual vs Historical Renewed Weightage")
    comparison = pd.DataFrame({
        "Factor": list(manual_weights.keys()),
        "Manual Weightage %": [manual_weights[k] for k in manual_weights],
        "Historical Renewed Weightage %": [history_weights.get(k, 0) for k in manual_weights],
        "Renewed Patient Average Score": [history_profile.get(k, np.nan) for k in manual_weights],
        "Historical Records Used": [history_valid_count.get(k, 0) for k in manual_weights],
    }).round(2)
    st.dataframe(comparison, use_container_width=True, hide_index=True)
    st.bar_chart(comparison.set_index("Factor")[["Manual Weightage %", "Historical Renewed Weightage %"]])

st.info(
    "Historical Renewed Pattern Score probability nahi hai. Ye current patient ka pattern, "
    "2026 mein actually renewed patients ke average engagement/outcome pattern se similarity dikhata hai."
)
