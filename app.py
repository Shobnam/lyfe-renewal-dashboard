import io
from datetime import date
from typing import List, Tuple

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Lyfe Renewal Intelligence", page_icon="🩺", layout="wide")
st.title("🩺 Lyfe Renewal Intelligence")
st.caption("Daily Excel upload → Fixed manual score + Logistic Regression → Final renewal call list")

CURRENT_SHEET = "Current_Renewal_Patients"
HISTORY_SHEET = "Historical_Renewed_2026"
OUTCOME_COLUMN = "Renewal Outcome"
FACTORS = ["LCM", "HC", "CT", "Psychotherapist", "Device/Vitals", "Clinical", "NPS"]
MODEL_FEATURES = [
    "LCM Score", "HC Score", "CT Score", "Psychotherapist Score",
    "Device/Vitals Score", "Clinical Score", "NPS Score",
    "Days Since Last Completed Call", "Total Relevant Vitals - Last 30 Days",
    "LCM Calls Completed", "HC Calls Completed", "CT Calls Completed",
    "Psychotherapist Sessions Completed",
]

# Fixed scoring settings. These are not shown as UI sliders.
MANUAL_WEIGHTS = {
    "LCM": 10,
    "HC": 10,
    "CT": 20,
    "Psychotherapist": 15,
    "Device/Vitals": 15,
    "Clinical": 20,
    "NPS": 10,
}
PENALTIES = {
    "Device Available But No Recent Vitals": 15,
    "Call Gap 31-60 Days": 2,
    "Call Gap 61-90 Days": 5,
    "Call Gap More Than 90 Days": 8,
    "No Completed Call": 10,
}
HIGH_THRESHOLD = 75
MODERATE_THRESHOLD = 50
LEARNING_RATE = 0.025
TRAINING_ITERATIONS = 2200
REGULARIZATION = 0.01
MANUAL_FINAL_WEIGHT = 0.50
LOGISTIC_FINAL_WEIGHT = 0.50
HIGH_VITALS_MIN = 10

DATE_COLUMNS = [
    "Subscription Start Date", "Subscription End Date", "Actual Renewal Date",
    "Next Plan Start Date", "Next Plan End Date", "Renewal Call 1 Date",
    "Renewal Call 2 Date", "Renewal Call 3 Date", "Last Renewal Call Date",
    "Last Completed Renewal Call Date", "Last Completed Call Date",
    "Last Auto Scheduled Call Date", "Last Auto Scheduled Completed Call Date",
    "Initial Active Weight Date", "Latest Active Weight Date", "Initial SBP Date",
    "Current SBP Date", "Initial LDL Date", "Current LDL Date",
    "Initial HbA1c Date", "Current HbA1c Date", "Latest NPS Date",
]
HIDDEN_TECHNICAL_COLUMNS = [
    "Vital Devices Available", "Devices With Recent Vitals",
    "Device Vital Coverage %", "Total Relevant Vitals - Last 30 Days",
]

# ---------------- Daily upload state ----------------
TODAY = date.today().isoformat()
for key, default in {
    "uploaded_excel_bytes": None,
    "uploaded_excel_name": None,
    "uploaded_excel_date": None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# The file expires automatically when the app sees a new calendar date.
if (
    st.session_state.uploaded_excel_bytes is not None
    and st.session_state.uploaded_excel_date != TODAY
):
    st.session_state.uploaded_excel_bytes = None
    st.session_state.uploaded_excel_name = None
    st.session_state.uploaded_excel_date = None
    if "renewal_excel_uploader" in st.session_state:
        del st.session_state["renewal_excel_uploader"]


@st.cache_data(show_spinner=False)
def read_workbook(file_bytes: bytes) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    buffer = io.BytesIO(file_bytes)
    xls = pd.ExcelFile(buffer, engine="openpyxl")
    missing = [s for s in [CURRENT_SHEET, HISTORY_SHEET] if s not in xls.sheet_names]
    if missing:
        raise ValueError(f"Missing Excel tabs: {missing}. Available tabs: {xls.sheet_names}")
    current = pd.read_excel(xls, sheet_name=CURRENT_SHEET)
    history = pd.read_excel(xls, sheet_name=HISTORY_SHEET)
    return current, history, xls.sheet_names


if st.session_state.uploaded_excel_bytes is None:
    uploaded = st.file_uploader(
        "📂 Aaj ka patients_data_2026 Excel upload karein",
        type=["xlsx"],
        key="renewal_excel_uploader",
    )
    if uploaded is None:
        st.info(f"Required tabs: `{CURRENT_SHEET}` and `{HISTORY_SHEET}`. File next calendar day automatically remove ho jayegi.")
        st.stop()
    st.session_state.uploaded_excel_bytes = uploaded.getvalue()
    st.session_state.uploaded_excel_name = uploaded.name
    st.session_state.uploaded_excel_date = TODAY
    st.rerun()
else:
    upload_col, date_col, remove_col = st.columns([4, 2, 1])
    upload_col.success(f"📂 Using: {st.session_state.uploaded_excel_name}")
    date_col.info(f"Upload date: {st.session_state.uploaded_excel_date}")
    if remove_col.button("🗑️ Remove", use_container_width=True):
        st.session_state.uploaded_excel_bytes = None
        st.session_state.uploaded_excel_name = None
        st.session_state.uploaded_excel_date = None
        read_workbook.clear()
        if "renewal_excel_uploader" in st.session_state:
            del st.session_state["renewal_excel_uploader"]
        st.rerun()

try:
    current, history, sheet_names = read_workbook(st.session_state.uploaded_excel_bytes)
except Exception as exc:
    st.error("Excel read nahi hua.")
    st.exception(exc)
    st.stop()

# ---------------- Helpers ----------------
def clean_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = df.columns.astype(str).str.strip()
    df = df.loc[:, ~df.columns.duplicated()].copy()
    if "Patient Name" not in df.columns and "Patient Name2" in df.columns:
        df = df.rename(columns={"Patient Name2": "Patient Name"})
    return df


def numeric_series(df: pd.DataFrame, col: str, default=np.nan) -> pd.Series:
    if col not in df.columns:
        return pd.Series(default, index=df.index, dtype=float)
    return pd.to_numeric(df[col], errors="coerce")


def first_numeric(df: pd.DataFrame, names: List[str], default=np.nan) -> pd.Series:
    out = pd.Series(np.nan, index=df.index, dtype=float)
    for name in names:
        if name in df.columns:
            out = out.fillna(pd.to_numeric(df[name], errors="coerce"))
    return out.fillna(default) if not pd.isna(default) else out


def yes_series(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(False, index=df.index)
    return df[col].astype(str).str.strip().str.upper().isin(["YES", "Y", "TRUE", "1"])


def service_component(df, scheduled, completed, pct, target):
    sch = numeric_series(df, scheduled, 0).fillna(0).astype(float)
    comp = numeric_series(df, completed, 0).fillna(0).astype(float)
    stored_pct = numeric_series(df, pct)
    calculated_pct = pd.Series(np.where(sch > 0, comp / sch * 100, np.nan), index=df.index)
    completion = stored_pct.fillna(calculated_pct).clip(0, 100)
    effort = (comp / float(target)).clip(0, 1) * 100
    return (completion * 0.70 + effort * 0.30).where(sch > 0)


def trend_score(series: pd.Series) -> pd.Series:
    text = series.astype(str).str.strip().str.upper()
    score = pd.Series(np.nan, index=series.index, dtype=float)
    missing = (
        text.eq("") | text.str.contains("NO INITIAL", na=False)
        | text.str.contains("NO CURRENT", na=False)
        | text.str.contains("NO ACTIVE", na=False)
        | text.str.contains("ONLY ONE", na=False)
    )
    score[text.str.contains("REDUCED|IMPROVED", regex=True, na=False)] = 100
    score[text.str.contains("NO CHANGE|SAME|STABLE", regex=True, na=False)] = 50
    score[text.str.contains("INCREASED|DETERIORATED", regex=True, na=False)] = 0
    score[missing] = np.nan
    return score


def build_components(df: pd.DataFrame, historical=False):
    c = pd.DataFrame(index=df.index)
    c["LCM"] = service_component(df, "LCM Calls Scheduled", "LCM Calls Completed", "LCM Completion %", 6)
    c["HC"] = service_component(df, "HC Calls Scheduled", "HC Calls Completed", "HC Completion %", 6)
    c["CT"] = service_component(df, "CT Calls Scheduled", "CT Calls Completed", "CT Completion %", 6)
    c["Psychotherapist"] = service_component(
        df, "Psychotherapist Sessions Scheduled", "Psychotherapist Sessions Completed",
        "Psychotherapist Completion %", 4,
    )
    available = first_numeric(df, ["Vital Devices Available", "Active Device Count"])
    inferred_available = (
        yes_series(df, "BP Monitor Available").astype(int)
        + yes_series(df, "Weight Scale Available").astype(int)
        + yes_series(df, "Glucometer Available").astype(int)
        + yes_series(df, "Oximeter Available").astype(int)
    ).astype(float)
    available = available.fillna(inferred_available)
    recent_vitals = first_numeric(
        df, ["Total Relevant Vitals - Last 30 Days", "Vital Readings - Last 30 Days of Plan"], 0
    ).fillna(0)
    coverage = first_numeric(df, ["Device Vital Coverage %"])
    devices_used = first_numeric(df, ["Devices With Recent Vitals"])
    inferred_coverage = pd.Series(np.where(available > 0, devices_used / available * 100, np.nan), index=df.index)
    coverage = coverage.fillna(inferred_coverage)
    activity = (recent_vitals / (available.clip(lower=1) * 4)).clip(0, 1) * 100
    c["Device/Vitals"] = (coverage.clip(0, 100).fillna(activity) * 0.80 + activity * 0.20).where(available > 0)
    trends = []
    for col in ["Weight Trend", "SBP Trend", "LDL Trend", "HbA1c Trend"]:
        if col in df.columns:
            trends.append(trend_score(df[col]).rename(col))
    if historical:
        for col in ["Latest SBP - Plan", "Latest LDL - Plan", "Latest HbA1c - Plan"]:
            if col in df.columns:
                values = pd.to_numeric(df[col], errors="coerce")
                trends.append(pd.Series(np.where(values.notna(), 50, np.nan), index=df.index, name=col))
    c["Clinical"] = pd.concat(trends, axis=1).mean(axis=1, skipna=True) if trends else np.nan
    c["NPS"] = numeric_series(df, "Latest NPS Score").clip(0, 10) * 10
    return c, available, recent_vitals


def weighted_score(components, weights):
    numerator = pd.Series(0.0, index=components.index)
    denominator = pd.Series(0.0, index=components.index)
    for factor, weight in weights.items():
        valid = components[factor].notna()
        numerator += components[factor].fillna(0) / 100 * weight
        denominator += valid.astype(float) * weight
    return (numerator / denominator.replace(0, np.nan) * 100).fillna(0)


def possibility(score):
    return pd.Series(np.select(
        [score >= HIGH_THRESHOLD, score >= MODERATE_THRESHOLD],
        ["HIGH", "MODERATE"], default="LOW"
    ), index=score.index)


def model_features(df, components):
    cols = [
        components["LCM"].fillna(0).clip(0, 100),
        components["HC"].fillna(0).clip(0, 100),
        components["CT"].fillna(0).clip(0, 100),
        components["Psychotherapist"].fillna(0).clip(0, 100),
        components["Device/Vitals"].fillna(0).clip(0, 100),
        components["Clinical"].fillna(0).clip(0, 100),
        components["NPS"].fillna(0).clip(0, 100),
        first_numeric(df, ["Days Since Last Completed Call", "Days Since Last Completed Call at Plan End"], 180).clip(0, 365),
        first_numeric(df, ["Total Relevant Vitals - Last 30 Days", "Vital Readings - Last 30 Days of Plan"], 0).clip(0, 500),
        numeric_series(df, "LCM Calls Completed", 0).fillna(0).clip(0, 50),
        numeric_series(df, "HC Calls Completed", 0).fillna(0).clip(0, 50),
        numeric_series(df, "CT Calls Completed", 0).fillna(0).clip(0, 50),
        numeric_series(df, "Psychotherapist Sessions Completed", 0).fillna(0).clip(0, 50),
    ]
    return np.column_stack([x.to_numpy(dtype=float) for x in cols])


def renewal_outcome(df):
    outcome = pd.Series(np.nan, index=df.index, dtype=float)
    if OUTCOME_COLUMN in df.columns:
        values = pd.to_numeric(df[OUTCOME_COLUMN], errors="coerce")
        outcome[values.isin([0, 1])] = values[values.isin([0, 1])]
    status_col = "Renewal Status" if "Renewal Status" in df.columns else "Renewal Outcome Label"
    if status_col in df.columns:
        status = df[status_col].astype(str).str.strip().str.upper()
        outcome[status.eq("RENEWED")] = 1
        outcome[status.isin(["NOT RENEWED", "EXPIRED", "LOST", "DROPPED"])] = 0
    return outcome


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -35, 35)))


def train_logistic(history_df, components):
    y_all = renewal_outcome(history_df)
    mask = y_all.isin([0, 1])
    if mask.sum() == 0:
        raise ValueError("Historical sheet mein valid Renewal Outcome 0/1 ya Renewal Status labels nahi mile.")
    X = model_features(history_df.loc[mask].reset_index(drop=True), components.loc[mask].reset_index(drop=True))
    y = y_all.loc[mask].to_numpy(dtype=float)
    renewed, not_renewed = int((y == 1).sum()), int((y == 0).sum())
    if renewed == 0 or not_renewed == 0:
        raise ValueError(f"Dono classes chahiye. Renewed={renewed}, Not Renewed={not_renewed}.")
    means, sds = X.mean(axis=0), X.std(axis=0)
    sds[sds < 0.000001] = 1.0
    Xs = (X - means) / sds
    beta = np.zeros(Xs.shape[1] + 1)
    rw, nw = len(y) / (2 * renewed), len(y) / (2 * not_renewed)
    sw = np.where(y == 1, rw, nw)
    for _ in range(TRAINING_ITERATIONS):
        error = (sigmoid(beta[0] + Xs @ beta[1:]) - y) * sw
        beta[0] -= LEARNING_RATE * error.mean()
        beta[1:] -= LEARNING_RATE * ((Xs.T @ error) / len(y) + REGULARIZATION * beta[1:])
    return {
        "coefficients": beta, "means": means, "sds": sds,
        "training_rows": len(y), "renewed_rows": renewed, "not_renewed_rows": not_renewed,
    }


def predict_logistic(df, components, model):
    X = model_features(df, components)
    Xs = (X - model["means"]) / model["sds"]
    return pd.Series(sigmoid(model["coefficients"][0] + Xs @ model["coefficients"][1:]) * 100, index=df.index)


def logistic_label(scores):
    return pd.Series(np.select(
        [scores >= 75, scores >= 50, scores >= 30],
        ["HIGH RENEWAL CHANCE", "MODERATE RENEWAL CHANCE", "LOW RENEWAL CHANCE"],
        default="VERY LOW RENEWAL CHANCE"
    ), index=scores.index)


def num(row, col, default=np.nan):
    value = pd.to_numeric(pd.Series([row.get(col)]), errors="coerce").iloc[0]
    return default if pd.isna(value) else value


def fmt_num(value):
    return f"{float(value):.1f}".rstrip("0").rstrip(".")


def positive_highlights(row):
    parts = []
    for label, completed_col, scheduled_col, minimum in [
        ("LCM", "LCM Calls Completed", "LCM Calls Scheduled", 2),
        ("HC", "HC Calls Completed", "HC Calls Scheduled", 2),
        ("CT", "CT Calls Completed", "CT Calls Scheduled", 3),
    ]:
        completed, scheduled = num(row, completed_col), num(row, scheduled_col)
        if pd.notna(completed) and pd.notna(scheduled) and scheduled > 0 and completed >= minimum:
            parts.append(f"{label}: {int(completed)}/{int(scheduled)} completed")
    for col, label, unit in [
        ("Weight Reduction (kg)", "Weight reduced", "kg"),
        ("SBP Reduction from Initial", "SBP reduced", "mmHg"),
        ("LDL Reduction from Initial", "LDL reduced", ""),
        ("HbA1c Reduction from Initial", "HbA1c reduced", ""),
    ]:
        value = num(row, col)
        if pd.notna(value) and value > 0:
            parts.append(f"{label} by {fmt_num(value)}{(' ' + unit) if unit else ''}")
    nps = num(row, "Latest NPS Score")
    if pd.notna(nps) and nps >= 9:
        parts.append(f"Promoter NPS {int(round(nps))}")
    return "; ".join(parts) if parts else "No strong positive signal available"


def risk_factors(row):
    parts = []
    gap = num(row, "Days Since Last Completed Call")
    if pd.isna(gap):
        parts.append("No completed-call date available")
    elif gap > 90:
        parts.append(f"Last completed call {int(round(gap))} days ago")
    elif gap > 60:
        parts.append(f"Call gap {int(round(gap))} days")
    devices = num(row, "Vital Devices Available", num(row, "Active Device Count", 0))
    vitals = num(row, "Total Relevant Vitals - Last 30 Days", 0)
    if devices > 0 and vitals == 0:
        parts.append("Device available but no recent vitals")
    pending = num(row, "System Renewal Calls Remaining", 0)
    if pending > 0:
        parts.append(f"{int(round(pending))} renewal call(s) pending")
    return "; ".join(parts) if parts else "No major risk flag"


def decide_caller(row):
    hc = str(row.get("HC Name", "") or "").strip()
    lcm = str(row.get("LCM Name", "") or "").strip()
    for n in [3, 2, 1]:
        status = str(row.get(f"Renewal Call {n} Status", "") or "").strip().upper()
        role = str(row.get(f"Renewal Call {n} Role", "") or "").strip().upper()
        if status == "COMPLETED":
            if role in ["COACH", "HC", "HEALTH COACH"] and hc:
                return hc
            if role == "LCM" and lcm:
                return lcm
    if num(row, "Total Relevant Vitals - Last 30 Days", 0) >= HIGH_VITALS_MIN and lcm:
        return lcm
    improvements = ["Weight Reduction (kg)", "SBP Reduction from Initial", "LDL Reduction from Initial", "HbA1c Reduction from Initial"]
    if any(num(row, c, 0) > 0 for c in improvements) and hc:
        return hc
    return hc or lcm or "Unassigned"


def determine_caller_role(row):
    caller = str(row.get("Caller", "") or "").strip()
    hc = str(row.get("HC Name", "") or "").strip()
    lcm = str(row.get("LCM Name", "") or "").strip()
    if caller and hc and caller.casefold() == hc.casefold():
        return "HC"
    if caller and lcm and caller.casefold() == lcm.casefold():
        return "LCM"
    for n in [3, 2, 1]:
        role = str(row.get(f"Renewal Call {n} Role", "") or "").strip().upper()
        if role in ["COACH", "HEALTH COACH"]:
            return "HC"
        if role in ["HC", "LCM"]:
            return role
    return "UNASSIGNED"


def suggested_pitch(row):
    name = str(row.get("Patient Name", "patient") or "patient").strip()
    plan = str(row.get("Current Plan Name", "current plan") or "current plan").strip()
    logistic = num(row, "Logistic Model Score", 0)
    guidance = (
        "Ask for renewal confirmation and next-plan preference." if logistic >= 75 else
        "Reinforce achieved benefits and resolve objections." if logistic >= 50 else
        "First understand barriers, affordability and service concerns."
    )
    return (
        f"Discuss renewal with {name}. Highlight {row.get('Positive Highlights', '')}. "
        f"Position renewal as continuity of {plan}, care-team support, monitoring and further improvement. {guidance}"
    )


def format_expiry_status(days):
    if pd.isna(days):
        return ""
    days = int(round(float(days)))
    if days < 0:
        return f"Plan expired {abs(days)} days ago"
    if days == 0:
        return "Plan expires today"
    return f"{days} days remaining"


def to_excel_bytes(df, model):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="Renewal_Call_List", index=False)
        pd.DataFrame({
            "Feature": ["Intercept"] + MODEL_FEATURES,
            "Coefficient": model["coefficients"],
            "Historical Mean": [np.nan] + list(model["means"]),
            "Historical SD": [np.nan] + list(model["sds"]),
        }).to_excel(writer, sheet_name="Logistic_Model", index=False)
        ws = writer.book["Renewal_Call_List"]
        for col_idx, name in enumerate(df.columns, 1):
            if "Date" in name:
                for row_idx in range(2, len(df) + 2):
                    ws.cell(row=row_idx, column=col_idx).number_format = "dd-mmm-yyyy"
        ws.freeze_panes = "A2"
        ws.auto_filter.ref = ws.dimensions
    return output.getvalue()

# ---------------- Clean and score ----------------
current, history = clean_df(current), clean_df(history)
if current.empty or history.empty:
    st.error("Current ya Historical sheet empty hai.")
    st.stop()

for df in [current, history]:
    for col in DATE_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

current_comp, current_devices, current_vitals = build_components(current)
history_comp, _, _ = build_components(history, historical=True)
try:
    model = train_logistic(history, history_comp)
except Exception as exc:
    st.error("Logistic Regression train nahi hua.")
    st.exception(exc)
    st.stop()

manual_base = weighted_score(current_comp, MANUAL_WEIGHTS)
gap = first_numeric(current, ["Days Since Last Completed Call"])
if gap.isna().all() and "Last Completed Call Date" in current.columns:
    gap = (pd.Timestamp.today().normalize() - current["Last Completed Call Date"].dt.normalize()).dt.days

call_penalty = pd.Series(0.0, index=current.index)
call_penalty[gap.between(31, 60, inclusive="both")] = PENALTIES["Call Gap 31-60 Days"]
call_penalty[gap.between(61, 90, inclusive="both")] = PENALTIES["Call Gap 61-90 Days"]
call_penalty[gap > 90] = PENALTIES["Call Gap More Than 90 Days"]
call_penalty[gap.isna()] = PENALTIES["No Completed Call"]
device_penalty = (
    ((current_devices > 0) & (current_vitals == 0)).astype(float)
    * PENALTIES["Device Available But No Recent Vitals"]
)
manual_score = (manual_base - call_penalty - device_penalty).clip(0, 100)
logistic_score = predict_logistic(current, current_comp, model)
final_score = (manual_score * MANUAL_FINAL_WEIGHT + logistic_score * LOGISTIC_FINAL_WEIGHT).clip(0, 100)

scored = current.copy()
scored["Manual Renewal Score"] = manual_score.round(1)
scored["Manual Possibility"] = possibility(manual_score)
scored["Logistic Model Score"] = logistic_score.round(1)
scored["Logistic Prediction"] = logistic_label(logistic_score)
scored["Final Renewal Score"] = final_score.round(1)
scored["Final Renewal Possibility"] = possibility(final_score)
scored["Call Gap Penalty"] = call_penalty
scored["Device Penalty"] = device_penalty
scored["Caller"] = scored.apply(decide_caller, axis=1)
scored["Caller Role"] = scored.apply(determine_caller_role, axis=1)
scored["Positive Highlights"] = scored.apply(positive_highlights, axis=1)
scored["Risk Factors"] = scored.apply(risk_factors, axis=1)
scored["Suggested Renewal Pitch"] = scored.apply(suggested_pitch, axis=1)
if "Days Remaining" in scored.columns:
    scored["Days to / Since Plan Expiry"] = scored["Days Remaining"].apply(format_expiry_status)

call_list = scored.copy()
if "Renewal Status" in call_list.columns:
    call_list = call_list[
        call_list["Renewal Status"].astype(str).str.strip().str.upper().ne("RENEWED")
    ]
call_list = call_list.sort_values("Final Renewal Score", ascending=False).reset_index(drop=True)

# ---------------- Top filters ----------------
st.markdown("### 🔎 Call List Filters")
filtered = call_list.copy()

filter_cols = st.columns(5)
filter_definitions = [
    ("Caller", "📞 Caller"),
    ("HC Name", "👤 HC"),
    ("LCM Name", "👤 LCM"),
    ("Caller Role", "🎭 Role"),
]
for box, (col, label) in zip(filter_cols[:4], filter_definitions):
    if col in filtered.columns:
        options = sorted([x for x in filtered[col].dropna().astype(str).unique() if x.strip()])
        selected = box.multiselect(label, options, key=f"top_filter_{col}")
        if selected:
            filtered = filtered[filtered[col].astype(str).isin(selected)]
    else:
        box.multiselect(label, [], disabled=True, key=f"top_filter_missing_{col}")

search_text = filter_cols[4].text_input("🔍 Identifier / Patient Name").strip().lower()
if search_text:
    identifier = filtered.get("Identifier", pd.Series("", index=filtered.index)).astype(str).str.lower()
    patient_name = filtered.get("Patient Name", pd.Series("", index=filtered.index)).astype(str).str.lower()
    filtered = filtered[
        identifier.str.contains(search_text, na=False, regex=False)
        | patient_name.str.contains(search_text, na=False, regex=False)
    ]

secondary_cols = st.columns(3)
for box, (col, label) in zip(secondary_cols, [
    ("Final Renewal Possibility", "Final Possibility"),
    ("Included Because", "Included Because"),
    ("Renewal Status", "Renewal Status"),
]):
    if col in filtered.columns:
        options = sorted([x for x in filtered[col].dropna().astype(str).unique() if x.strip()])
        selected = box.multiselect(label, options, key=f"top_filter_{col}")
        if selected:
            filtered = filtered[filtered[col].astype(str).isin(selected)]
    else:
        box.multiselect(label, [], disabled=True, key=f"top_filter_missing_{col}")

# ---------------- Dashboard ----------------
metrics = st.columns(6)
metrics[0].metric("Call List", len(filtered))
metrics[1].metric("Final High", int(filtered["Final Renewal Possibility"].eq("HIGH").sum()))
metrics[2].metric("Final Moderate", int(filtered["Final Renewal Possibility"].eq("MODERATE").sum()))
metrics[3].metric("Final Low", int(filtered["Final Renewal Possibility"].eq("LOW").sum()))
metrics[4].metric("Renewed Training", model["renewed_rows"])
metrics[5].metric("Not-Renewed Training", model["not_renewed_rows"])

important_columns = [
    "Identifier", "Patient Name", "Caller", "Caller Role", "HC Name", "LCM Name",
    "Manual Renewal Score", "Manual Possibility", "Logistic Model Score", "Logistic Prediction",
    "Final Renewal Score", "Final Renewal Possibility", "Positive Highlights", "Risk Factors",
    "Suggested Renewal Pitch", "Email", "Subscription Start Date", "Subscription End Date",
    "Days to / Since Plan Expiry", "Included Because", "Renewal Status",
]
remaining = [
    c for c in filtered.columns
    if c not in important_columns
    and c not in HIDDEN_TECHNICAL_COLUMNS
    and c != "Days Remaining"
]
display_columns = [c for c in important_columns if c in filtered.columns] + remaining
filtered_display = filtered.copy()
for col in DATE_COLUMNS:
    if col in filtered_display.columns:
        filtered_display[col] = pd.to_datetime(
            filtered_display[col], errors="coerce"
        ).dt.strftime("%d-%b-%Y").fillna("")

tab1, tab2, tab3, tab4 = st.tabs([
    "📞 Renewal Call List", "👤 Patient Drill-down", "🧠 Logistic Model", "📊 Component Scores"
])

with tab1:
    st.subheader("Renewal Call List")
    st.dataframe(filtered_display[display_columns], use_container_width=True, hide_index=True, height=620)
    c1, c2 = st.columns(2)
    c1.download_button(
        "📥 Download filtered CSV",
        filtered_display[display_columns].to_csv(index=False).encode("utf-8-sig"),
        "renewal_call_list.csv",
        "text/csv",
    )
    export_cols = [c for c in display_columns if c in call_list.columns]
    c2.download_button(
        "📥 Download full Excel",
        to_excel_bytes(call_list[export_cols], model),
        "renewal_call_list_scored.xlsx",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

with tab2:
    st.subheader("Patient-level details")
    if filtered_display.empty:
        st.info("Current filters mein patient nahi hai.")
    else:
        choices = filtered_display.apply(
            lambda r: f"{r.get('Identifier', '')} | {r.get('Patient Name', '')}", axis=1
        ).tolist()
        selected_patient = st.selectbox("Patient select karein", choices)
        row = filtered_display.iloc[choices.index(selected_patient)]
        a, b, c, d = st.columns(4)
        a.metric("Final Score", row.get("Final Renewal Score", ""))
        b.metric("Manual Score", row.get("Manual Renewal Score", ""))
        c.metric("Logistic Score", row.get("Logistic Model Score", ""))
        d.metric("Caller", row.get("Caller", ""))
        st.markdown(f"**Caller Role:** {row.get('Caller Role', '')}")
        st.markdown(f"**Positive Highlights:** {row.get('Positive Highlights', '')}")
        st.markdown(f"**Risk Factors:** {row.get('Risk Factors', '')}")
        st.markdown(f"**Suggested Renewal Pitch:** {row.get('Suggested Renewal Pitch', '')}")
        sections = {
            "Plan & Renewal": [x for x in row.index if any(k in x for k in ["Subscription", "Plan", "Renewal", "Included Because"])],
            "Calls": [x for x in row.index if any(k in x for k in ["Calls Scheduled", "Calls Completed", "Completion %", "Last Completed Call"])],
            "Devices & Vitals": [x for x in row.index if any(k in x for k in ["Device", "Vitals", "Readings - Last 30 Days", "Vital Engagement"])],
            "Clinical": [x for x in row.index if any(k in x for k in ["Weight", "SBP", "LDL", "HbA1c"])],
            "NPS": [x for x in row.index if "NPS" in x or "Feedback" in x],
        }
        for section, cols in sections.items():
            with st.expander(section, expanded=(section == "Plan & Renewal")):
                st.dataframe(
                    pd.DataFrame({"Field": cols, "Value": [row.get(x, "") for x in cols]}),
                    use_container_width=True,
                    hide_index=True,
                )

with tab3:
    st.subheader("Logistic Regression Model")
    st.write(
        f"Training rows: **{model['training_rows']}** | Renewed: **{model['renewed_rows']}** | "
        f"Not Renewed: **{model['not_renewed_rows']}**"
    )
    coef_df = pd.DataFrame({
        "Feature": ["Intercept"] + MODEL_FEATURES,
        "Coefficient": model["coefficients"],
        "Historical Mean": [np.nan] + list(model["means"]),
        "Historical SD": [np.nan] + list(model["sds"]),
    })
    coef_df["Direction"] = np.where(
        coef_df["Coefficient"] > 0.0001,
        "POSITIVE",
        np.where(coef_df["Coefficient"] < -0.0001, "NEGATIVE", "NEUTRAL"),
    )
    st.dataframe(coef_df.round(6), use_container_width=True, hide_index=True)
    z_values = np.arange(-6, 6.01, 0.25)
    st.line_chart(
        pd.DataFrame({"Z Score": z_values, "Logistic Probability": sigmoid(z_values)}).set_index("Z Score")
    )

with tab4:
    st.subheader("Current-patient component scores")
    component_view = current_comp.copy().round(1)
    if "Identifier" in current.columns:
        component_view.insert(0, "Identifier", current["Identifier"].values)
    if "Patient Name" in current.columns:
        component_view.insert(1, "Patient Name", current["Patient Name"].values)
    st.dataframe(component_view, use_container_width=True, hide_index=True)

st.info(
    "Manual weight, penalties aur thresholds fixed hain. Filters top par hain. "
    "Uploaded Excel next calendar day app rerun hote hi automatically remove ho jayegi. "
    "Final Score = 50% Manual + 50% Logistic."
)
