import io
import json
import os
import smtplib
import tempfile
from datetime import date, datetime
from email.mime.application import MIMEApplication
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.utils import formataddr, parseaddr

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
    "LCM Score",
    "HC Score",
    "CT Score",
    "Psychotherapist Score",
    "Device/Vitals Score",
    "Clinical Score",
    "NPS Score",
    "Days Since Last Completed Call",
    "Total Relevant Vitals - Last 30 Days",
    "LCM Calls Completed",
    "HC Calls Completed",
    "CT Calls Completed",
    "Psychotherapist Sessions Completed",
]
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
HIGH_THRESHOLD, MODERATE_THRESHOLD = 75, 50
LEARNING_RATE, TRAINING_ITERATIONS, REGULARIZATION = 0.025, 2200, 0.01
MANUAL_FINAL_WEIGHT, LOGISTIC_FINAL_WEIGHT = 0.50, 0.50
HIGH_VITALS_MIN = 10
SMTP_SERVER, SMTP_PORT = "smtp.sendgrid.net", 587
DEFAULT_REPORT_SENDER = "shobnambegum@lupindigitalhealth.com"
DEFAULT_REPORT_RECIPIENTS = "operationsteam@lupindigitalhealth.com"
DEFAULT_REPORT_BCC = "shobnambegum@lupindigitalhealth.com"
DATE_COLUMNS = [
    "Subscription Start Date",
    "Subscription End Date",
    "Actual Renewal Date",
    "Next Plan Start Date",
    "Next Plan End Date",
    "Renewal Call 1 Date",
    "Renewal Call 2 Date",
    "Renewal Call 3 Date",
    "Last Renewal Call Date",
    "Last Completed Renewal Call Date",
    "Last Completed Call Date",
    "Initial Active Weight Date",
    "Latest Active Weight Date",
    "Initial SBP Date",
    "Current SBP Date",
    "Initial LDL Date",
    "Current LDL Date",
    "Initial HbA1c Date",
    "Current HbA1c Date",
    "Latest NPS Date",
]


def clean_df(df):
    df = df.copy()
    df.columns = df.columns.astype(str).str.strip()
    df = df.loc[:, ~df.columns.duplicated()].copy()
    if "Patient Name" not in df.columns and "Patient Name2" in df.columns:
        df = df.rename(columns={"Patient Name2": "Patient Name"})
    return df


def normalize_identifier(s):
    return s.fillna("").astype(str).str.strip().str.upper().str.replace(r"\.0$", "", regex=True)


def numeric_series(df, col, default=np.nan):
    return pd.to_numeric(df[col], errors="coerce") if col in df.columns else pd.Series(default, index=df.index, dtype=float)


def first_numeric(df, names, default=np.nan):
    out = pd.Series(np.nan, index=df.index, dtype=float)
    for name in names:
        if name in df.columns:
            out = out.fillna(pd.to_numeric(df[name], errors="coerce"))
    return out.fillna(default) if not pd.isna(default) else out


def yes_series(df, col):
    return (
        df[col].astype(str).str.strip().str.upper().isin(["YES", "Y", "TRUE", "1"])
        if col in df.columns
        else pd.Series(False, index=df.index)
    )


def service_component(df, scheduled, completed, pct, target):
    sch = numeric_series(df, scheduled, 0).fillna(0)
    comp = numeric_series(df, completed, 0).fillna(0)
    calculated = pd.Series(np.where(sch > 0, comp / sch * 100, np.nan), index=df.index)
    completion = numeric_series(df, pct).fillna(calculated).clip(0, 100)
    effort = (comp / float(target)).clip(0, 1) * 100
    return (completion * 0.70 + effort * 0.30).where(sch > 0)


def trend_score(s):
    text = s.astype(str).str.strip().str.upper()
    score = pd.Series(np.nan, index=s.index, dtype=float)
    score[text.str.contains("REDUCED|IMPROVED", regex=True, na=False)] = 100
    score[text.str.contains("NO CHANGE|SAME|STABLE", regex=True, na=False)] = 50
    score[text.str.contains("INCREASED|DETERIORATED", regex=True, na=False)] = 0
    score[text.str.contains("NO INITIAL|NO CURRENT|NO ACTIVE|ONLY ONE", regex=True, na=False) | text.eq("")] = np.nan
    return score


def build_components(df, historical=False):
    c = pd.DataFrame(index=df.index)
    c["LCM"] = service_component(df, "LCM Calls Scheduled", "LCM Calls Completed", "LCM Completion %", 6)
    c["HC"] = service_component(df, "HC Calls Scheduled", "HC Calls Completed", "HC Completion %", 6)
    c["CT"] = service_component(df, "CT Calls Scheduled", "CT Calls Completed", "CT Completion %", 6)
    c["Psychotherapist"] = service_component(
        df, "Psychotherapist Sessions Scheduled", "Psychotherapist Sessions Completed", "Psychotherapist Completion %", 4
    )
    available = first_numeric(df, ["Vital Devices Available", "Active Device Count"])
    inferred = (
        yes_series(df, "BP Monitor Available").astype(int)
        + yes_series(df, "Weight Scale Available").astype(int)
        + yes_series(df, "Glucometer Available").astype(int)
        + yes_series(df, "Oximeter Available").astype(int)
    )
    available = available.fillna(inferred.astype(float))
    recent = first_numeric(df, ["Total Relevant Vitals - Last 30 Days", "Vital Readings - Last 30 Days of Plan"], 0).fillna(0)
    coverage = first_numeric(df, ["Device Vital Coverage %"])
    used = first_numeric(df, ["Devices With Recent Vitals"])
    coverage = coverage.fillna(pd.Series(np.where(available > 0, used / available * 100, np.nan), index=df.index))
    activity = (recent / (available.clip(lower=1) * 4)).clip(0, 1) * 100
    c["Device/Vitals"] = (coverage.clip(0, 100).fillna(activity) * 0.8 + activity * 0.2).where(available > 0)
    trends = [trend_score(df[x]).rename(x) for x in ["Weight Trend", "SBP Trend", "LDL Trend", "HbA1c Trend"] if x in df.columns]
    c["Clinical"] = pd.concat(trends, axis=1).mean(axis=1, skipna=True) if trends else np.nan
    c["NPS"] = numeric_series(df, "Latest NPS Score").clip(0, 10) * 10
    return c, available, recent


def weighted_score(components, weights):
    total = pd.Series(0.0, index=components.index)
    for factor, weight in weights.items():
        total += components[factor].fillna(0).clip(0, 100) / 100 * weight
    return total / sum(weights.values()) * 100


def model_features(df, c):
    cols = (
        [c[x].fillna(0).clip(0, 100) for x in FACTORS]
        + [
            first_numeric(df, ["Days Since Last Completed Call", "Days Since Last Completed Call at Plan End"], 180).clip(0, 365),
            first_numeric(df, ["Total Relevant Vitals - Last 30 Days", "Vital Readings - Last 30 Days of Plan"], 0).clip(0, 500),
            numeric_series(df, "LCM Calls Completed", 0).fillna(0).clip(0, 50),
            numeric_series(df, "HC Calls Completed", 0).fillna(0).clip(0, 50),
            numeric_series(df, "CT Calls Completed", 0).fillna(0).clip(0, 50),
            numeric_series(df, "Psychotherapist Sessions Completed", 0).fillna(0).clip(0, 50),
        ]
    )
    return np.column_stack([x.to_numpy(float) for x in cols])


def renewal_outcome(df):
    out = pd.Series(np.nan, index=df.index, dtype=float)
    if OUTCOME_COLUMN in df.columns:
        v = pd.to_numeric(df[OUTCOME_COLUMN], errors="coerce")
        out[v.isin([0, 1])] = v[v.isin([0, 1])]
    col = "Renewal Status" if "Renewal Status" in df.columns else "Renewal Outcome Label"
    if col in df.columns:
        s = df[col].astype(str).str.strip().str.upper()
        out[s.eq("RENEWED")] = 1
        out[s.isin(["NOT RENEWED", "EXPIRED", "LOST", "DROPPED"])] = 0
    return out


def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -35, 35)))


def train_logistic(df, c):
    y0 = renewal_outcome(df)
    mask = y0.isin([0, 1])
    X = model_features(df.loc[mask].reset_index(drop=True), c.loc[mask].reset_index(drop=True))
    y = y0.loc[mask].to_numpy(float)
    r, n = int((y == 1).sum()), int((y == 0).sum())
    if not r or not n:
        raise ValueError(f"Both outcome classes required. Renewed={r}, Not Renewed={n}")
    mean, sd = X.mean(0), X.std(0)
    sd[sd < 1e-6] = 1
    xs = (X - mean) / sd
    beta = np.zeros(xs.shape[1] + 1)
    sw = np.where(y == 1, len(y) / (2 * r), len(y) / (2 * n))
    for _ in range(TRAINING_ITERATIONS):
        e = (sigmoid(beta[0] + xs @ beta[1:]) - y) * sw
        beta[0] -= LEARNING_RATE * e.mean()
        beta[1:] -= LEARNING_RATE * ((xs.T @ e) / len(y) + REGULARIZATION * beta[1:])
    return {
        "coefficients": beta,
        "means": mean,
        "sds": sd,
        "training_rows": len(y),
        "renewed_rows": r,
        "not_renewed_rows": n,
    }


def predict(df, c, m):
    x = model_features(df, c)
    return pd.Series(sigmoid(m["coefficients"][0] + ((x - m["means"]) / m["sds"]) @ m["coefficients"][1:]) * 100, index=df.index)


def possibility(s):
    return pd.Series(np.select([s >= 75, s >= 50], ["HIGH", "MODERATE"], default="LOW"), index=s.index)


def logistic_label(s):
    return pd.Series(
        np.select([s >= 75, s >= 50, s >= 30], ["HIGH RENEWAL CHANCE", "MODERATE RENEWAL CHANCE", "LOW RENEWAL CHANCE"], default="VERY LOW RENEWAL CHANCE"),
        index=s.index,
    )


def num(row, col, default=np.nan):
    v = pd.to_numeric(pd.Series([row.get(col)]), errors="coerce").iloc[0]
    return default if pd.isna(v) else v


def decide_caller(row):
    """Choose the caller using the established renewal ownership rules."""
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
    improvement_cols = [
        "Weight Reduction (kg)",
        "SBP Reduction from Initial",
        "LDL Reduction from Initial",
        "HbA1c Reduction from Initial",
    ]
    if any(num(row, col, 0) > 0 for col in improvement_cols) and hc:
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
    return "UNASSIGNED"


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
            parts.append(f"{label} by {float(value):.1f}{(' ' + unit) if unit else ''}")
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
    pending_calls = num(row, "System Renewal Calls Remaining", 0)
    if pending_calls > 0:
        parts.append(f"{int(round(pending_calls))} renewal call(s) pending")
    return "; ".join(parts) if parts else "No major risk flag"


def suggested_pitch(row):
    name = str(row.get("Patient Name", "patient") or "patient").strip()
    plan = str(row.get("Current Plan Name", "current plan") or "current plan").strip()
    logistic = num(row, "Logistic Model Score", 0)
    guidance = (
        "Ask for renewal confirmation and next-plan preference."
        if logistic >= 75
        else "Reinforce achieved benefits and resolve objections."
        if logistic >= 50
        else "First understand barriers, affordability and service concerns."
    )
    return (
        f"Discuss renewal with {name}. Highlight {row.get('Positive Highlights', '')}. "
        f"Position renewal as continuity of {plan}, care-team support, monitoring and further improvement. {guidance}"
    )


def prepare_reason_data(df):
    if df.empty:
        return pd.DataFrame(columns=["_Patient_Key", "Reason", "Reason Comment", "Reason Completion Time"])
    df = clean_df(df)
    lm = {c.lower().strip(): c for c in df.columns}
    pick = lambda a: next((lm[x] for x in a if x in lm), None)
    p = pick(["patient id", "patientid", "identifier", "patient identifier"])
    r = pick(["primary reason", "primary reason for non-renewal", "reason", "renewal reason"])
    if not p or not r:
        return pd.DataFrame(columns=["_Patient_Key", "Reason", "Reason Comment", "Reason Completion Time"])
    cm = pick(["comment", "comments", "remark", "remarks"])
    tm = pick(["completion time", "completed time", "response time", "submitted at"])
    out = pd.DataFrame(
        {
            "_Patient_Key": normalize_identifier(df[p]),
            "Reason": df[r].fillna("").astype(str).str.strip(),
            "Reason Comment": df[cm].fillna("").astype(str).str.strip() if cm else "",
            "Reason Completion Time": pd.to_datetime(df[tm], errors="coerce") if tm else pd.NaT,
        }
    )
    return out[out["_Patient_Key"].ne("")].sort_values("Reason Completion Time", na_position="first").drop_duplicates("_Patient_Key", keep="last")


def prepare_escalation(df, source):
    cols = [
        "_Patient_Key",
        "Escalation Source",
        "Escalation Date",
        "Owner",
        "Team",
        "Bucket",
        "Sub-Bucket",
        "Issue Type",
        "Type Of Issue",
        "Findings",
        "Ops Analysis",
        "Escalation Raised By",
        "Action Status",
        "Status",
        "LCM",
        "HC",
    ]
    if df.empty:
        return pd.DataFrame(columns=cols)
    df = clean_df(df)
    lm = {c.lower().strip(): c for c in df.columns}
    pick = lambda a: next((lm[x.lower()] for x in a if x.lower() in lm), None)
    p = pick(["Patient Id", "Patient ID", "Identifier", "Patient Identifier"])
    if not p:
        return pd.DataFrame(columns=cols)
    out = pd.DataFrame(index=df.index)
    out["_Patient_Key"] = normalize_identifier(df[p])
    out["Escalation Source"] = source
    d = pick(["Esc_Date", "Esc Date", "Escalation Date", "Date"])
    out["Escalation Date"] = pd.to_datetime(df[d], errors="coerce") if d else pd.NaT
    mapping = {
        "Owner": ["Owner"],
        "Team": ["Team"],
        "Bucket": ["Bucket"],
        "Sub-Bucket": ["Sub-Bucket", "Sub Bucket"],
        "Issue Type": ["Issue Type"],
        "Type Of Issue": ["Type Of Issue"],
        "Findings": ["Findings", "Finding"],
        "Ops Analysis": ["Ops Analysis", "Ops Analy", "Ops Analy:"],
        "Escalation Raised By": ["Escalation raised by"],
        "Action Status": ["Action Status"],
        "Status": ["Status"],
        "LCM": ["LCM"],
        "HC": ["HC"],
    }
    for k, a in mapping.items():
        x = pick(a)
        out[k] = df[x].fillna("").astype(str).str.strip() if x else ""
    return out[out["_Patient_Key"].ne("")][cols]


def is_valid_email(value):
    parsed = parseaddr(str(value or "").strip())[1]
    return bool(parsed and "@" in parsed and "." in parsed.split("@")[-1])


def parse_email_list(value):
    parts = str(value or "").replace(";", ",").split(",")
    return list(dict.fromkeys(x.strip() for x in parts if is_valid_email(x.strip())))


def connect_sendgrid(api_key):
    server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT, timeout=60)
    server.ehlo()
    server.starttls()
    server.ehlo()
    server.login("apikey", api_key.strip())
    return server


def renewal_report_excel(call_df, pending_df, received_df):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        call_df.drop(columns=["_Patient_Key"], errors="ignore").to_excel(writer, sheet_name="Renewal_Call_List", index=False)
        pending_df.drop(columns=["_Patient_Key"], errors="ignore").to_excel(writer, sheet_name="Reason_Pending", index=False)
        received_df.drop(columns=["_Patient_Key"], errors="ignore").to_excel(writer, sheet_name="Reason_Received", index=False)
        for ws in writer.book.worksheets:
            ws.freeze_panes = "A2"
            ws.auto_filter.ref = ws.dimensions
    return output.getvalue()


def build_report_html(call_df, pending_df, received_df, renewed_count):
    high = int(call_df.get("Final Renewal Possibility", pd.Series(dtype=str)).eq("HIGH").sum())
    moderate = int(call_df.get("Final Renewal Possibility", pd.Series(dtype=str)).eq("MODERATE").sum())
    low = int(call_df.get("Final Renewal Possibility", pd.Series(dtype=str)).eq("LOW").sum())
    escalation = int(call_df.get("Escalation Status", pd.Series(dtype=str)).eq("Escalation").sum())
    generated = datetime.now().strftime("%d-%b-%Y %I:%M %p")
    return f"""<html><body style='font-family:Arial;background:#f5f7fb;padding:20px'>
<div style='background:white;padding:24px;border-radius:12px;max-width:900px;margin:auto'>
<h2 style='color:#b4235a'>Lyfe Renewal Intelligence Report</h2><p>Generated: {generated}</p>
<table cellpadding='10' cellspacing='5'><tr><td><b>{len(call_df)}</b><br>Calling List</td><td><b>{renewed_count}</b><br>Renewed</td><td><b>{high}</b><br>High</td><td><b>{moderate}</b><br>Moderate</td><td><b>{low}</b><br>Low</td><td><b>{escalation}</b><br>Escalation</td><td><b>{len(pending_df)}</b><br>Reason Pending</td></tr></table>
<p><b>Action required:</b> {len(pending_df)} patient(s) have at least 3 completed renewal calls and no non-renewal reason.</p>
<p>The attached Excel contains Renewal Call List, Reason Pending, and Reason Received sheets.</p>
<p>Regards,<br>Lyfe Renewal Intelligence</p></div></body></html>"""


def create_report_email(sender, recipients, cc_list, bcc_list, subject, html_body, attachment_bytes, attachment_name):
    message = MIMEMultipart("mixed")
    message["From"] = formataddr(("Lyfe Renewal Intelligence", sender))
    message["To"] = ", ".join(recipients)
    if cc_list:
        message["Cc"] = ", ".join(cc_list)
    if bcc_list:
        message["Bcc"] = ", ".join(bcc_list)
    message["Subject"] = subject
    alternative = MIMEMultipart("alternative")
    alternative.attach(MIMETex

On Thu, 26 Nov 2026, 1:55 pm Shobnam Begum, <Shobnambegum@lupindigitalhealth.com> wrote:
import io
import smtplib
from datetime import date, datetime
from email.mime.application import MIMEApplication
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.utils import formataddr, parseaddr
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
MODEL_FEATURES = ["LCM Score", "HC Score", "CT Score", "Psychotherapist Score", "Device/Vitals Score", "Clinical Score", "NPS Score", "Days Since Last Completed Call", "Total Relevant Vitals - Last 30 Days", "LCM Calls Completed", "HC Calls Completed", "CT Calls Completed", "Psychotherapist Sessions Completed"]
MANUAL_WEIGHTS = {"LCM": 10, "HC": 10, "CT": 20, "Psychotherapist": 15, "Device/Vitals": 15, "Clinical": 20, "NPS": 10}
PENALTIES = {"Device Available But No Recent Vitals": 15, "Call Gap 31-60 Days": 2, "Call Gap 61-90 Days": 5, "Call Gap More Than 90 Days": 8, "No Completed Call": 10}
HIGH_THRESHOLD, MODERATE_THRESHOLD = 75, 50
LEARNING_RATE, TRAINING_ITERATIONS, REGULARIZATION = 0.025, 2200, 0.01
MANUAL_FINAL_WEIGHT, LOGISTIC_FINAL_WEIGHT = 0.50, 0.50
HIGH_VITALS_MIN = 10
SMTP_SERVER, SMTP_PORT = "smtp.sendgrid.net", 587
DEFAULT_REPORT_SENDER = "shobnambegum@lupindigitalhealth.com"
DEFAULT_REPORT_RECIPIENTS = "operationsteam@lupindigitalhealth.com"
DEFAULT_REPORT_BCC = "shobnambegum@lupindigitalhealth.com"
DATE_COLUMNS = ["Subscription Start Date", "Subscription End Date", "Actual Renewal Date", "Next Plan Start Date", "Next Plan End Date", "Renewal Call 1 Date", "Renewal Call 2 Date", "Renewal Call 3 Date", "Last Renewal Call Date", "Last Completed Renewal Call Date", "Last Completed Call Date", "Initial Active Weight Date", "Latest Active Weight Date", "Initial SBP Date", "Current SBP Date", "Initial LDL Date", "Current LDL Date", "Initial HbA1c Date", "Current HbA1c Date", "Latest NPS Date"]

def clean_df(df):
df = df.copy(); df.columns = df.columns.astype(str).str.strip(); df = df.loc[:, ~df.columns.duplicated()].copy()
if "Patient Name" not in df.columns and "Patient Name2" in df.columns: df = df.rename(columns={"Patient Name2": "Patient Name"})
return df

def normalize_identifier(s):
return s.fillna("").astype(str).str.strip().str.upper().str.replace(r".0$", "", regex=True)

def numeric_series(df, col, default=np.nan):
return pd.to_numeric(df[col], errors="coerce") if col in df.columns else pd.Series(default, index=df.index, dtype=float)

def first_numeric(df, names, default=np.nan):
out = pd.Series(np.nan, index=df.index, dtype=float)
for name in names:
if name in df.columns: out = out.fillna(pd.to_numeric(df[name], errors="coerce"))
return out.fillna(default) if not pd.isna(default) else out

def yes_series(df, col):
return df[col].astype(str).str.strip().str.upper().isin(["YES", "Y", "TRUE", "1"]) if col in df.columns else pd.Series(False, index=df.index)

def service_component(df, scheduled, completed, pct, target):
sch, comp = numeric_series(df, scheduled, 0).fillna(0), numeric_series(df, completed, 0).fillna(0)
calculated = pd.Series(np.where(sch > 0, comp / sch * 100, np.nan), index=df.index)
completion = numeric_series(df, pct).fillna(calculated).clip(0, 100)
effort = (comp / float(target)).clip(0, 1) * 100
return (completion * .70 + effort * .30).where(sch > 0)

def trend_score(s):
text = s.astype(str).str.strip().str.upper(); score = pd.Series(np.nan, index=s.index, dtype=float)
score[text.str.contains("REDUCED|IMPROVED", regex=True, na=False)] = 100
score[text.str.contains("NO CHANGE|SAME|STABLE", regex=True, na=False)] = 50
score[text.str.contains("INCREASED|DETERIORATED", regex=True, na=False)] = 0
score[text.str.contains("NO INITIAL|NO CURRENT|NO ACTIVE|ONLY ONE", regex=True, na=False) | text.eq("")] = np.nan
return score

def build_components(df, historical=False):
c = pd.DataFrame(index=df.index)
c["LCM"] = service_component(df, "LCM Calls Scheduled", "LCM Calls Completed", "LCM Completion %", 6)
c["HC"] = service_component(df, "HC Calls Scheduled", "HC Calls Completed", "HC Completion %", 6)
c["CT"] = service_component(df, "CT Calls Scheduled", "CT Calls Completed", "CT Completion %", 6)
c["Psychotherapist"] = service_component(df, "Psychotherapist Sessions Scheduled", "Psychotherapist Sessions Completed", "Psychotherapist Completion %", 4)
available = first_numeric(df, ["Vital Devices Available", "Active Device Count"])
inferred = yes_series(df, "BP Monitor Available").astype(int) + yes_series(df, "Weight Scale Available").astype(int) + yes_series(df, "Glucometer Available").astype(int) + yes_series(df, "Oximeter Available").astype(int)
available = available.fillna(inferred.astype(float))
recent = first_numeric(df, ["Total Relevant Vitals - Last 30 Days", "Vital Readings - Last 30 Days of Plan"], 0).fillna(0)
coverage = first_numeric(df, ["Device Vital Coverage %"])
used = first_numeric(df, ["Devices With Recent Vitals"])
coverage = coverage.fillna(pd.Series(np.where(available > 0, used / available * 100, np.nan), index=df.index))
activity = (recent / (available.clip(lower=1) * 4)).clip(0, 1) * 100
c["Device/Vitals"] = (coverage.clip(0, 100).fillna(activity) * .8 + activity * .2).where(available > 0)
trends = [trend_score(df[x]).rename(x) for x in ["Weight Trend", "SBP Trend", "LDL Trend", "HbA1c Trend"] if x in df.columns]
c["Clinical"] = pd.concat(trends, axis=1).mean(axis=1, skipna=True) if trends else np.nan
c["NPS"] = numeric_series(df, "Latest NPS Score").clip(0, 10) * 10
return c, available, recent

def weighted_score(components, weights):
total = pd.Series(0.0, index=components.index)
for factor, weight in weights.items(): total += components[factor].fillna(0).clip(0, 100) / 100 * weight
return total / sum(weights.values()) * 100

def model_features(df, c):
cols = [c[x].fillna(0).clip(0, 100) for x in FACTORS] + [first_numeric(df, ["Days Since Last Completed Call", "Days Since Last Completed Call at Plan End"], 180).clip(0,365), first_numeric(df, ["Total Relevant Vitals - Last 30 Days", "Vital Readings - Last 30 Days of Plan"], 0).clip(0,500), numeric_series(df,"LCM Calls Completed",0).fillna(0).clip(0,50), numeric_series(df,"HC Calls Completed",0).fillna(0).clip(0,50), numeric_series(df,"CT Calls Completed",0).fillna(0).clip(0,50), numeric_series(df,"Psychotherapist Sessions Completed",0).fillna(0).clip(0,50)]
return np.column_stack([x.to_numpy(float) for x in cols])

def renewal_outcome(df):
out = pd.Series(np.nan, index=df.index, dtype=float)
if OUTCOME_COLUMN in df.columns:
v = pd.to_numeric(df[OUTCOME_COLUMN], errors="coerce"); out[v.isin([0,1])] = v[v.isin([0,1])]
col = "Renewal Status" if "Renewal Status" in df.columns else "Renewal Outcome Label"
if col in df.columns:
s = df[col].astype(str).str.strip().str.upper(); out[s.eq("RENEWED")] = 1; out[s.isin(["NOT RENEWED","EXPIRED","LOST","DROPPED"])] = 0
return out

def sigmoid(z): return 1/(1+np.exp(-np.clip(z,-35,35)))

def train_logistic(df,c):
y0=renewal_outcome(df); mask=y0.isin([0,1])
X=model_features(df.loc[mask].reset_index(drop=True),c.loc[mask].reset_index(drop=True)); y=y0.loc[mask].to_numpy(float)
r,n=int((y==1).sum()),int((y==0).sum())
if not r or not n: raise ValueError(f"Both outcome classes required. Renewed={r}, Not Renewed={n}")
mean,sd=X.mean(0),X.std(0); sd[sd<1e-6]=1; xs=(X-mean)/sd; beta=np.zeros(xs.shape[1]+1); sw=np.where(y==1,len(y)/(2r),len(y)/(2n))
for _ in range(TRAINING_ITERATIONS):
e=(sigmoid(beta[0]+xs@beta[1:])-y)sw; beta[0]-=LEARNING_RATEe.mean(); beta[1:]-=LEARNING_RATE((xs.T@e)/len(y)+REGULARIZATIONbeta[1:])
return {"coefficients":beta,"means":mean,"sds":sd,"training_rows":len(y),"renewed_rows":r,"not_renewed_rows":n}

def predict(df,c,m):
x=model_features(df,c); return pd.Series(sigmoid(m["coefficients"][0]+((x-m["means"])/m["sds"])@m["coefficients"][1:])100,index=df.index)

def possibility(s): return pd.Series(np.select([s>=75,s>=50],["HIGH","MODERATE"],default="LOW"),index=s.index)
def logistic_label(s): return pd.Series(np.select([s>=75,s>=50,s>=30],["HIGH RENEWAL CHANCE","MODERATE RENEWAL CHANCE","LOW RENEWAL CHANCE"],default="VERY LOW RENEWAL CHANCE"),index=s.index)
def num(row,col,default=np.nan):
v=pd.to_numeric(pd.Series([row.get(col)]),errors="coerce").iloc[0]; return default if pd.isna(v) else v

def decide_caller(row):
"""Choose the caller using the established renewal ownership rules."""
hc = str(row.get("HC Name", "") or "").strip()
lcm = str(row.get("LCM Name", "") or "").strip()
# First preference: owner role of the latest completed renewal call.
for n in [3, 2, 1]:
status = str(row.get(f"Renewal Call {n} Status", "") or "").strip().upper()
role = str(row.get(f"Renewal Call {n} Role", "") or "").strip().upper()
if status == "COMPLETED":
if role in ["COACH", "HC", "HEALTH COACH"] and hc:
return hc
if role == "LCM" and lcm:
return lcm
# High recent vital engagement goes to LCM when available.
if num(row, "Total Relevant Vitals - Last 30 Days", 0) >= HIGH_VITALS_MIN and lcm:
return lcm
# Clinical improvement goes to HC when available.
improvement_cols = [
"Weight Reduction (kg)",
"SBP Reduction from Initial",
"LDL Reduction from Initial",
"HbA1c Reduction from Initial",
]
if any(num(row, col, 0) > 0 for col in improvement_cols) and hc:
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
return "UNASSIGNED"

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
parts.append(f"{label} by {float(value):.1f}{(' ' + unit) if unit else ''}")
nps = num(row, "Latest NPS Score")
if pd.notna(nps) and nps >= 9:
parts.append(f"Promoter NPS {int(round(nps))}")
return "; ".join(parts) if parts else "No strong positive signal available"

def risk_factors(row):
parts = []
gap = num(row, "Days Since Last Completed Call")
if pd.isna(gap): parts.append("No completed-call date available")
elif gap > 90: parts.append(f"Last completed call {int(round(gap))} days ago")
elif gap > 60: parts.append(f"Call gap {int(round(gap))} days")
devices = num(row, "Vital Devices Available", num(row, "Active Device Count", 0))
vitals = num(row, "Total Relevant Vitals - Last 30 Days", 0)
if devices > 0 and vitals == 0: parts.append("Device available but no recent vitals")
pending_calls = num(row, "System Renewal Calls Remaining", 0)
if pending_calls > 0: parts.append(f"{int(round(pending_calls))} renewal call(s) pending")
return "; ".join(parts) if parts else "No major risk flag"

def suggested_pitch(row):
name = str(row.get("Patient Name", "patient") or "patient").strip()
plan = str(row.get("Current Plan Name", "current plan") or "current plan").strip()
logistic = num(row, "Logistic Model Score", 0)
guidance = ("Ask for renewal confirmation and next-plan preference." if logistic >= 75 else
"Reinforce achieved benefits and resolve objections." if logistic >= 50 else
"First understand barriers, affordability and service concerns.")
return (f"Discuss renewal with {name}. Highlight {row.get('Positive Highlights', '')}. "
f"Position renewal as continuity of {plan}, care-team support, monitoring and further improvement. {guidance}")

def prepare_reason_data(df):
if df.empty:return pd.DataFrame(columns=["_Patient_Key","Reason","Reason Comment","Reason Completion Time"])
df=clean_df(df); lm={c.lower().strip():c for c in df.columns}
pick=lambda a: next((lm[x] for x in a if x in lm),None)
p=pick(["patient id","patientid","identifier","patient identifier"]); r=pick(["primary reason","primary reason for non-renewal","reason","renewal reason"])
if not p or not r:return pd.DataFrame(columns=["_Patient_Key","Reason","Reason Comment","Reason Completion Time"])
cm=pick(["comment","comments","remark","remarks"]); tm=pick(["completion time","completed time","response time","submitted at"])
out=pd.DataFrame({"_Patient_Key":normalize_identifier(df[p]),"Reason":df[r].fillna("").astype(str).str.strip(),"Reason Comment":df[cm].fillna("").astype(str).str.strip() if cm else "","Reason Completion Time":pd.to_datetime(df[tm],errors="coerce") if tm else pd.NaT})
return out[out["_Patient_Key"].ne("")].sort_values("Reason Completion Time",na_position="first").drop_duplicates("_Patient_Key",keep="last")

def prepare_escalation(df,source):
cols=["_Patient_Key","Escalation Source","Escalation Date","Owner","Team","Bucket","Sub-Bucket","Issue Type","Type Of Issue","Findings","Ops Analysis","Escalation Raised By","Action Status","Status","LCM","HC"]
if df.empty:return pd.DataFrame(columns=cols)
df=clean_df(df); lm={c.lower().strip():c for c in df.columns}; pick=lambda a:next((lm[x.lower()] for x in a if x.lower() in lm),None)
p=pick("Patient Id","Patient ID","Identifier","Patient Identifier")
if not p:return pd.DataFrame(columns=cols)
out=pd.DataFrame(index=df.index);out["_Patient_Key"]=normalize_identifier(df[p]);out["Escalation Source"]=source
d=pick("Esc_Date","Esc Date","Escalation Date","Date");out["Escalation Date"]=pd.to_datetime(df[d],errors="coerce") if d else pd.NaT
mapping={"Owner":["Owner"],"Team":["Team"],"Bucket":["Bucket"],"Sub-Bucket":["Sub-Bucket","Sub Bucket"],"Issue Type":["Issue Type"],"Type Of Issue":["Type Of Issue"],"Findings":["Findings","Finding"],"Ops Analysis":["Ops Analysis","Ops Analy","Ops Analy:"],"Escalation Raised By":["Escalation raised by"],"Action Status":["Action Status"],"Status":["Status"],"LCM":["LCM"],"HC":["HC"]}
for k,a in mapping.items():
x=pick(a);out[k]=df[x].fillna("").astype(str).str.strip() if x else ""
return out[out["_Patient_Key"].ne("")][cols]

def is_valid_email(value):
parsed = parseaddr(str(value or "").strip())[1]
return bool(parsed and "@" in parsed and "." in parsed.split("@")[-1])

def parse_email_list(value):
parts = str(value or "").replace(";", ",").split(",")
return list(dict.fromkeys(x.strip() for x in parts if is_valid_email(x.strip())))

def connect_sendgrid(api_key):
server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT, timeout=60)
server.ehlo(); server.starttls(); server.ehlo()
server.login("apikey", api_key.strip())
return server

def renewal_report_excel(call_df, pending_df, received_df):
output = io.BytesIO()
with pd.ExcelWriter(output, engine="openpyxl") as writer:
call_df.drop(columns=["_Patient_Key"], errors="ignore").to_excel(writer, sheet_name="Renewal_Call_List", index=False)
pending_df.drop(columns=["_Patient_Key"], errors="ignore").to_excel(writer, sheet_name="Reason_Pending", index=False)
received_df.drop(columns=["_Patient_Key"], errors="ignore").to_excel(writer, sheet_name="Reason_Received", index=False)
for ws in writer.book.worksheets:
ws.freeze_panes = "A2"
ws.auto_filter.ref = ws.dimensions
return output.getvalue()

def build_report_html(call_df, pending_df, received_df, renewed_count):
high = int(call_df.get("Final Renewal Possibility", pd.Series(dtype=str)).eq("HIGH").sum())
moderate = int(call_df.get("Final Renewal Possibility", pd.Series(dtype=str)).eq("MODERATE").sum())
low = int(call_df.get("Final Renewal Possibility", pd.Series(dtype=str)).eq("LOW").sum())
escalation = int(call_df.get("Escalation Status", pd.Series(dtype=str)).eq("Escalation").sum())
generated = datetime.now().strftime("%d-%b-%Y %I:%M %p")
return f"""<html><body style='font-family:Arial;background:#f5f7fb;padding:20px'>
<div style='background:white;padding:24px;border-radius:12px;max-width:900px;margin:auto'>
<h2 style='color:#b4235a'>Lyfe Renewal Intelligence Report</h2><p>Generated: {generated}</p>
<table cellpadding='10' cellspacing='5'><tr><td><b>{len(call_df)}</b><br>Calling List</td><td><b>{renewed_count}</b><br>Renewed</td><td><b>{high}</b><br>High</td><td><b>{moderate}</b><br>Moderate</td><td><b>{low}</b><br>Low</td><td><b>{escalation}</b><br>Escalation</td><td><b>{len(pending_df)}</b><br>Reason Pending</td></tr></table>
<p><b>Action required:</b> {len(pending_df)} patient(s) have at least 3 completed renewal calls and no non-renewal reason.</p>
<p>The attached Excel contains Renewal Call List, Reason Pending, and Reason Received sheets.</p>
<p>Regards,<br>Lyfe Renewal Intelligence</p></div></body></html>"""

def create_report_email(sender, recipients, cc_list, bcc_list, subject, html_body, attachment_bytes, attachment_name):
message = MIMEMultipart("mixed")
message["From"] = formataddr(("Lyfe Renewal Intelligence", sender))
message["To"] = ", ".join(recipients)
if cc_list: message["Cc"] = ", ".join(cc_list)
if bcc_list: message["Bcc"] = ", ".join(bcc_list)
message["Subject"] = subject
alternative = MIMEMultipart("alternative")
alternative.attach(MIMEText("The detailed renewal report is attached.", "plain", "utf-8"))
alternative.attach(MIMEText(html_body, "html", "utf-8"))
message.attach(alternative)
attachment = MIMEApplication(attachment_bytes, _subtype="vnd.openxmlformats-officedocument.spreadsheetml.sheet")
attachment.add_header("Content-Disposition", "attachment", filename=attachment_name)
message.attach(attachment)
return message
TODAY=date.today().isoformat()
for k,v in {"uploaded_excel_bytes":None,"uploaded_excel_name":None,"uploaded_excel_date":None}.items():
if k not in st.session_state:st.session_state[k]=v
if st.session_state.uploaded_excel_bytes is not None and st.session_state.uploaded_excel_date!=TODAY:
st.session_state.uploaded_excel_bytes=None;st.session_state.uploaded_excel_name=None;st.session_state.uploaded_excel_date=None
if st.session_state.uploaded_excel_bytes is None:
up=st.file_uploader("📂 Upload today's patients_data_2026 Excel",type=["xlsx"])
if up is None:st.info(f"Required: {CURRENT_SHEET}, {HISTORY_SHEET}. Optional: Reasons, Escalation Whatsapp, Escalation Mail");st.stop()
st.session_state.uploaded_excel_bytes=up.getvalue();st.session_state.uploaded_excel_name=up.name;st.session_state.uploaded_excel_date=TODAY;st.rerun()
else:
a,b,c=st.columns([4,2,1]);a.success(f"📂 {st.session_state.uploaded_excel_name}");b.info(f"Upload date: {TODAY}")
if c.button("🗑️ Remove"):st.session_state.uploaded_excel_bytes=None;st.rerun()
try:
xls=pd.ExcelFile(io.BytesIO(st.session_state.uploaded_excel_bytes),engine="openpyxl")
current=pd.read_excel(xls,CURRENT_SHEET);history=pd.read_excel(xls,HISTORY_SHEET);lookup={x.lower().strip():x for x in xls.sheet_names}
whats=pd.read_excel(xls,lookup["escalation whatsapp"]) if "escalation whatsapp" in lookup else pd.DataFrame();mail=pd.read_excel(xls,lookup["escalation mail"]) if "escalation mail" in lookup else pd.DataFrame();reason=pd.DataFrame()
for sh in xls.sheet_names:
if sh.lower().strip() in [CURRENT_SHEET.lower(),HISTORY_SHEET.lower(),"escalation whatsapp","escalation mail"]:continue
t=pd.read_excel(xls,sh);names={str(c).lower().strip() for c in t.columns}
if names.intersection({"patient id","patientid","identifier","patient identifier"}) and names.intersection({"primary reason","primary reason for non-renewal","reason","renewal reason"}):reason=t;break
except Exception as e:st.exception(e);st.stop()
current,history=clean_df(current),clean_df(history)
for df in [current,history]:
for col in DATE_COLUMNS:
if col in df.columns:df[col]=pd.to_datetime(df[col],errors="coerce")
escalations=pd.concat([prepare_escalation(whats,"WhatsApp"),prepare_escalation(mail,"Mail")],ignore_index=True)
reasons=prepare_reason_data(reason)
cc,devices,vitals=build_components(current);hc,_,_=build_components(history,True)
try:model=train_logistic(history,hc)
except Exception as e:st.error("Logistic model training failed");st.exception(e);st.stop()
base=weighted_score(cc,MANUAL_WEIGHTS);gap=first_numeric(current,["Days Since Last Completed Call","Days Since Last Completed Call at Plan End"])
if gap.isna().all() and "Last Completed Call Date" in current.columns:gap=(pd.Timestamp.today().normalize()-pd.to_datetime(current["Last Completed Call Date"],errors="coerce").dt.normalize()).dt.days
gap=gap.clip(lower=0);cp=pd.Series(0.,index=current.index);cp[gap.between(31,60)]=2;cp[gap.between(61,90)]=5;cp[gap>90]=8;cp[gap.isna()]=10
dp=((devices>0)&(vitals<=0)).astype(float)15;manual=(base-cp-dp).clip(0,100);logistic=predict(current,cc,model).clip(0,100);final=(manual.5+logistic.5).clip(0,100)
scored=current.copy()
for f in FACTORS:scored[f"{f} Component Score"]=cc[f].round(1)
scored["Manual Base Score Before Penalty"]=base.round(1);scored["Call Gap Penalty"]=cp;scored["Device Penalty"]=dp;scored["Manual Renewal Score"]=manual.round(1);scored["Manual Possibility"]=possibility(manual);scored["Logistic Model Score"]=logistic.round(1);scored["Logistic Prediction"]=logistic_label(logistic);scored["Final Renewal Score"]=final.round(1);scored["Final Renewal Possibility"]=possibility(final)
scored["Caller"] = scored.apply(decide_caller, axis=1)
scored["Caller Role"] = scored.apply(determine_caller_role, axis=1)
scored["Positive Highlights"] = scored.apply(positive_highlights, axis=1)
scored["Risk Factors"] = scored.apply(risk_factors, axis=1)
scored["Suggested Renewal Pitch"] = scored.apply(suggested_pitch, axis=1)
scored["_Patient_Key"]=normalize_identifier(scored.get("Identifier",pd.Series("",index=scored.index)));scored=scored.merge(reasons,on="_Patient_Key",how="left");scored["Reason"]=scored["Reason"].fillna("").astype(str).str.strip()
sys=first_numeric(scored,["System Renewal Calls Completed","Renewal Calls Completed"]);statuses=[c for c in ["Renewal Call 1 Status","Renewal Call 2 Status","Renewal Call 3 Status"] if c in scored.columns];fallback=sum(scored[c].fillna("").astype(str).str.upper().eq("COMPLETED").astype(int) for c in statuses) if statuses else pd.Series(0,index=scored.index)
scored["Renewal Calls Completed Count"]=sys.fillna(fallback).fillna(0).astype(int)
notrenewed=scored["Renewal Status"].fillna("").astype(str).str.upper().ne("RENEWED") if "Renewal Status" in scored.columns else pd.Series(True,index=scored.index)
eligible=scored[notrenewed].copy();received=eligible[eligible["Reason"].ne("")].copy();call_list=eligible[eligible["Reason"].eq("")].copy();call_list["Reason Status"]=np.where(call_list["Renewal Calls Completed Count"]>=3,"REASON PENDING","");pending=call_list[call_list["Renewal Calls Completed Count"]>=3].copy()
keys=set(escalations["_Patient_Key"].dropna().astype(str));call_list["Escalation Status"]=np.where(call_list["_Patient_Key"].isin(keys),"Escalation","No Escalation");pending["Escalation Status"]=np.where(pending["_Patient_Key"].isin(keys),"Escalation","No Escalation")
st.markdown("### 🔎 Call List Filters");filtered=call_list.copy();boxes=st.columns(7)
for box,(col,label) in zip(boxes[:6],[("Source","📌 Source"),("Escalation Status","🚨 Escalation"),("Caller","📞 Caller"),("HC Name","HC"),("LCM Name","LCM"),("Final Renewal Possibility","Possibility")]):
opts=sorted(filtered[col].dropna().astype(str).unique()) if col in filtered.columns else [];sel=box.multiselect(label,opts)
if sel:filtered=filtered[filtered[col].astype(str).isin(sel)]
q=boxes[6].text_input("Identifier / Name").lower().strip()
if q:filtered=filtered[filtered.get("Identifier","").astype(str).str.lower().str.contains(q,regex=False)|filtered.get("Patient Name","").astype(str).str.lower().str.contains(q,regex=False)]
m=st.columns(6);m[0].metric("Call List",len(filtered));m[1].metric("High",int(filtered["Final Renewal Possibility"].eq("HIGH").sum()));m[2].metric("Moderate",int(filtered["Final Renewal Possibility"].eq("MODERATE").sum()));m[3].metric("Low",int(filtered["Final Renewal Possibility"].eq("LOW").sum()));m[4].metric("Escalation",int(filtered["Escalation Status"].eq("Escalation").sum()));m[5].metric("No Escalation",int(filtered["Escalation Status"].eq("No Escalation").sum()))
t1,t2,t3,t4,t5,t6=st.tabs(["📞 Renewal Call List","⏳ Reason Pending","👤 Patient Drill-down","🧠 Logistic Model","📊 Component Scores","✉️ Email Report"])
with t1:
st.dataframe(filtered.sort_values("Final Renewal Score",ascending=False),use_container_width=True,hide_index=True,height=620)
st.download_button("Download CSV",filtered.drop(columns=["_Patient_Key"],errors="ignore").to_csv(index=False).encode("utf-8-sig"),"renewal_call_list.csv")
with t2:
st.metric("Reason Pending",len(pending));st.dataframe(pending,use_container_width=True,hide_index=True)
with t3:
mode=st.radio("🚨 Escalation Filter",["All Patients","Escalation Patients","No Escalation Patients"],horizontal=True);d=filtered.copy()
if mode=="Escalation Patients":d=d[d["Escalation Status"].eq("Escalation")]
if mode=="No Escalation Patients":d=d[d["Escalation Status"].eq("No Escalation")]
if d.empty:st.info("No patient found")
else:
choices=d.apply(lambda r:f"{r.get('Identifier','')} | {r.get('Patient Name','')} | {r.get('Escalation Status','')}",axis=1).tolist();choice=st.selectbox("Select patient",choices);row=d.iloc[choices.index(choice)]
a,b,c,e,f=st.columns(5);a.metric("Manual",row.get("Manual Renewal Score"));b.metric("Logistic",row.get("Logistic Model Score"));c.metric("Final",row.get("Final Renewal Score"));e.metric("Caller",row.get("Caller",""));f.metric("Escalation",row.get("Escalation Status"))
st.markdown("### 📞 Renewal Call Talking Points")
st.markdown(f"Positive Highlights: {row.get('Positive Highlights','')}")
st.markdown(f"Risk Factors: {row.get('Risk Factors','')}")
st.info(f"💬 Suggested Renewal Pitch: {row.get('Suggested Renewal Pitch','')}")
st.markdown("### 📋 Patient context")
detail_sections = {
"📅 Plan & Renewal": [
"Source", "Email", "Caller", "Caller Role", "HC Name", "LCM Name",
"Current Plan Name", "Current Plan Duration", "Current Plan Price",
"Subscription Start Date", "Subscription End Date", "Days Remaining",
"Days to / Since Plan Expiry", "Renewal Status", "Actual Renewal Date",
"Next Plan Name", "Next Plan Start Date", "Next Plan End Date",
"System Renewal Calls Scheduled", "System Renewal Calls Completed",
"System Renewal Calls Remaining", "System Renewal Follow Up Status",
"Renewal Call 1 Date", "Renewal Call 1 Status", "Renewal Call 1 Subject", "Renewal Call 1 Role",
"Renewal Call 2 Date", "Renewal Call 2 Status", "Renewal Call 2 Subject", "Renewal Call 2 Role",
"Renewal Call 3 Date", "Renewal Call 3 Status", "Renewal Call 3 Subject", "Renewal Call 3 Role",
"Last Renewal Call Date", "Last Completed Renewal Call Date",
],
"📞 Service Calls": [
"Last Completed Call Date", "Days Since Last Completed Call",
"LCM Calls Scheduled", "LCM Calls Completed", "LCM Completion %",
"HC Calls Scheduled", "HC Calls Completed", "HC Completion %",
"CT Calls Scheduled", "CT Calls Completed", "CT Completion %", "CT Physio Name",
"Psychotherapist Sessions Scheduled", "Psychotherapist Sessions Completed",
"Psychotherapist Completion %", "Psychotherapist Engagement",
],
"⚖️ Weight Details": [
"Initial Active Weight (kg)", "Initial Active Weight Date", "Initial Active Weight Status",
"Latest Active Weight (kg)", "Latest Active Weight Date", "Latest Active Weight Status",
"Total Active Weight Readings - Subscription", "Weight Reduction (kg)",
"Weight Change %", "Weight Trend",
],
"🩺 SBP Details": [
"Initial SBP", "Initial SBP Date", "Initial SBP Status",
"Current SBP", "Current SBP Date", "Current SBP Status",
"SBP Reduction from Initial", "SBP Change %", "SBP Trend",
],
"🧪 LDL Details": [
"Initial LDL", "Initial LDL Date", "Current LDL", "Current LDL Date",
"LDL Reduction from Initial", "LDL Change %", "LDL Trend",
],
"🩸 HbA1c Details": [
"Initial HbA1c", "Initial HbA1c Date", "Current HbA1c", "Current HbA1c Date",
"HbA1c Reduction from Initial", "HbA1c Change %", "HbA1c Trend",
],
"📟 Devices & Vitals": [
"Active Device Count", "Active Devices", "BP Monitor Available", "Weight Scale Available",
"Glucometer Available", "Oximeter Available", "BP Readings - Last 30 Days",
"Weight Readings - Last 30 Days", "Glucose Readings - Last 30 Days",
"SpO2 Readings - Last 30 Days", "Heart Rate Readings - Last 30 Days",
"Total Relevant Vitals - Last 30 Days", "Last Vital Date", "Days Since Last Vital",
"Vital Devices Available", "Devices With Recent Vitals", "Device Vital Coverage %",
"Vital Engagement Status",
],
"⭐ NPS & Feedback": [
"Latest NPS Score", "Latest NPS Category", "Latest NPS Date",
"Latest NPS Comments", "Latest NPS Data Source", "Average NPS Score",
"Total NPS Responses", "Promoter Responses", "Passive Responses", "Detractor Responses",
"Overall Service Feedback", "Health Coach Feedback", "LCM Feedback",
"Cardio Therapy Feedback", "Meditation Feedback", "Monitoring Devices Feedback",
"Lyfe App Feedback", "Alert Management Feedback",
],
}
for section_name, section_fields in detail_sections.items():
visible_fields = [field for field in section_fields if field in row.index]
if visible_fields:
with st.expander(section_name, expanded=(section_name == "📅 Plan & Renewal")):
st.dataframe(
pd.DataFrame({
"Field": visible_fields,
"Value": [row.get(field, "") for field in visible_fields],
}),
use_container_width=True,
hide_index=True,
)
with st.expander("🧮 Scoring Breakdown",expanded=False):
fields=["Manual Base Score Before Penalty","Call Gap Penalty","Device Penalty","Manual Renewal Score","Logistic Model Score","Final Renewal Score"]+[f"{x} Component Score" for x in FACTORS]
st.dataframe(pd.DataFrame({"Field":fields,"Value":[row.get(x,"") for x in fields]}),use_container_width=True,hide_index=True)
pk=normalize_identifier(pd.Series([row.get("Identifier","")])).iloc[0];pe=escalations[escalations["_Patient_Key"].eq(pk)].copy()
with st.expander(f"🚨 Escalations ({len(pe)})",expanded=not pe.empty):
if pe.empty:st.info("No escalation record found")
else:st.dataframe(pe.drop(columns=["_Patient_Key"]),use_container_width=True,hide_index=True)
with t4:
coef=pd.DataFrame({"Feature":["Intercept"]+MODEL_FEATURES,"Coefficient":model["coefficients"],"Mean":[np.nan]+list(model["means"]),"SD":[np.nan]+list(model["sds"])});st.write(f"Rows: {model['training_rows']} | Renewed: {model['renewed_rows']} | Not renewed: {model['not_renewed_rows']}");st.dataframe(coef,use_container_width=True,hide_index=True)
with t5:
cv=cc.round(1);cv.insert(0,"Identifier",current.get("Identifier",pd.Series("",index=current.index)));cv.insert(1,"Patient Name",current.get("Patient Name",pd.Series("",index=current.index)));st.dataframe(cv,use_container_width=True,hide_index=True)
with t6:
st.subheader("✉️ Send Renewal Intelligence Report")
renewed_count = int((~notrenewed).sum())
report_html = build_report_html(call_list, pending, received, renewed_count)
report_attachment = renewal_report_excel(call_list, pending, received)
p1,p2,p3,p4,p5=st.columns(5)
p1.metric("Calling List",len(call_list));p2.metric("Renewed",renewed_count)
p3.metric("High",int(call_list["Final Renewal Possibility"].eq("HIGH").sum()))
p4.metric("Escalation",int(call_list["Escalation Status"].eq("Escalation").sum()))
p5.metric("Reason Pending",len(pending))
with st.expander("📧 Report Preview",expanded=True):
st.components.v1.html(report_html,height=500,scrolling=True)
left,right=st.columns(2)
with left:
report_sender=st.text_input("Verified Sender Email",value=DEFAULT_REPORT_SENDER,key="report_sender").strip()
report_to=st.text_input("To",value=DEFAULT_REPORT_RECIPIENTS,key="report_to")
report_cc=st.text_input("CC (optional)",value="",key="report_cc")
report_bcc=st.text_input("BCC",value=DEFAULT_REPORT_BCC,key="report_bcc")
with right:
default_subject=f"Lyfe Renewal Report | {renewed_count} Renewed | {len(call_list)} Calling | {len(pending)} Reason Pending | {datetime.now().strftime('%d-%b-%Y')}"
report_subject=st.text_input("Subject",value=default_subject,key="report_subject")
try: secret_key=st.secrets.get("SENDGRID_API_KEY","")
except Exception: secret_key=""
report_api_key=st.text_input("SendGrid API Key",value=secret_key,type="password",key="report_sendgrid_key",help="Best practice: save SENDGRID_API_KEY in Streamlit Secrets.")
test_connection=st.button("🔐 Test SendGrid Connection",use_container_width=True,key="test_sendgrid")
if test_connection:
server=None
try:
if not is_valid_email(report_sender): raise ValueError("Verified Sender Email is invalid.")
if not report_api_key.strip(): raise ValueError("SendGrid API Key is required.")
server=connect_sendgrid(report_api_key);st.success("✅ SendGrid connection successful.")
except Exception as exc: st.error(f"❌ SendGrid connection failed: {exc}")
finally:
if server:
try: server.quit()
except Exception: pass
recipients=parse_email_list(report_to);cc_list=parse_email_list(report_cc);bcc_list=parse_email_list(report_bcc)
confirmation=st.checkbox(f"I confirm this report should be sent to {len(recipients)} recipient(s).",key="confirm_report")
send_report=st.button("🚀 Send Renewal Report",type="primary",use_container_width=True,disabled=(not confirmation or not recipients or not report_api_key.strip() or not is_valid_email(report_sender)),key="send_report")
if send_report:
server=None
try:
message=create_report_email(report_sender,recipients,cc_list,bcc_list,report_subject.strip() or default_subject,report_html,report_attachment,f"lyfe_renewal_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx")
server=connect_sendgrid(report_api_key);all_recipients=list(dict.fromkeys(recipients+cc_list+bcc_list));server.sendmail(report_sender,all_recipients,message.as_string())
st.success(f"✅ Renewal report sent successfully to {', '.join(recipients)}.")
except Exception as exc: st.error("❌ Report could not be sent.");st.code(str(exc))
finally:
if server:
try: server.quit()
except Exception: pass
st.download_button("📥 Download Same Report Excel",report_attachment,f"lyfe_renewal_report_{datetime.now().strftime('%Y%m%d')}.xlsx","application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",use_container_width=True)
st.info("Manual score uses a fixed 100-point denominator. Missing components receive zero. Final Score = 50% Manual + 50% Logistic.")
