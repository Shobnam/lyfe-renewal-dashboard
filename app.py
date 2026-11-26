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
        np.select(
            [s >= 75, s >= 50, s >= 30],
            ["HIGH RENEWAL CHANCE", "MODERATE RENEWAL CHANCE", "LOW RENEWAL CHANCE"],
            default="VERY LOW RENEWAL CHANCE",
        ),
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
    return (
        out[out["_Patient_Key"].ne("")]
        .sort_values("Reason Completion Time", na_position="first")
        .drop_duplicates("_Patient_Key", keep="last")
    )


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
    alternative.attach(MIMEText("The detailed renewal report is attached.", "plain", "utf-8"))
    alternative.attach(MIMEText(html_body, "html", "utf-8"))
    message.attach(alternative)
    attachment = MIMEApplication(attachment_bytes, _subtype="vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    attachment.add_header("Content-Disposition", "attachment", filename=attachment_name)
    message.attach(attachment)
    return message


TODAY = date.today().isoformat()
TEMP_DIR = os.path.join(tempfile.gettempdir(), "lyfe_renewal_cache")
os.makedirs(TEMP_DIR, exist_ok=True)
META_PATH = os.path.join(TEMP_DIR, "upload_meta.json")
EXCEL_PATH = os.path.join(TEMP_DIR, "daily_patients_data.xlsx")


def load_cached_upload():
    if os.path.exists(META_PATH) and os.path.exists(EXCEL_PATH):
        try:
            with open(META_PATH, "r", encoding="utf-8") as f:
                meta = json.load(f)
            if meta.get("date") == TODAY:
                with open(EXCEL_PATH, "rb") as f:
                    return f.read(), meta.get("name", "patients_data_2026.xlsx")
        except Exception:
            pass
    return None, None


def save_cached_upload(file_bytes, filename):
    with open(EXCEL_PATH, "wb") as f:
        f.write(file_bytes)
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump({"date": TODAY, "name": filename}, f)


def clear_cached_upload():
    for path in (META_PATH, EXCEL_PATH):
        if os.path.exists(path):
            os.remove(path)


excel_bytes, excel_name = load_cached_upload()

if excel_bytes is None:
    up = st.file_uploader("📂 Upload today's patients_data_2026 Excel", type=["xlsx"])
    if up is None:
        st.info(f"Required: {CURRENT_SHEET}, {HISTORY_SHEET}. Optional: Reasons, Escalation Whatsapp, Escalation Mail")
        st.stop()
    save_cached_upload(up.getvalue(), up.name)
    st.rerun()
else:
    a, b, c = st.columns([4, 2, 1])
    a.success(f"📂 {excel_name}")
    b.info(f"Upload date: {TODAY}")
    if c.button("🗑️ Remove"):
        clear_cached_upload()
        st.rerun()

try:
    xls = pd.ExcelFile(io.BytesIO(excel_bytes), engine="openpyxl")
    current = pd.read_excel(xls, CURRENT_SHEET)
    history = pd.read_excel(xls, HISTORY_SHEET)
    lookup = {x.lower().strip(): x for x in xls.sheet_names}
    whats = pd.read_excel(xls, lookup["escalation whatsapp"]) if "escalation whatsapp" in lookup else pd.DataFrame()
    mail = pd.read_excel(xls, lookup["escalation mail"]) if "escalation mail" in lookup else pd.DataFrame()
    reason = pd.DataFrame()
    for sh in xls.sheet_names:
        if sh.lower().strip() in [CURRENT_SHEET.lower(), HISTORY_SHEET.lower(), "escalation whatsapp", "escalation mail"]:
            continue
        t = pd.read_excel(xls, sh)
        names = {str(c).lower().strip() for c in t.columns}
        if names.intersection({"patient id", "patientid", "identifier", "patient identifier"}) and names.intersection(
            {"primary reason", "primary reason for non-renewal", "reason", "renewal reason"}
        ):
            reason = t
            break

    current = clean_df(current)
    history = clean_df(history)

    for df in (current, history):
        for col in DATE_COLUMNS:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce")

    curr_comp, curr_avail, curr_recent = build_components(current, historical=False)
    hist_comp, _, _ = build_components(history, historical=True)

    current_manual = weighted_score(curr_comp, MANUAL_WEIGHTS)
    gap_days = first_numeric(current, ["Days Since Last Completed Call", "Days Since Last Completed Call at Plan End"], 180)
    penalty_pts = pd.Series(0.0, index=current.index)
    penalty_pts += np.where((curr_avail > 0) & (curr_recent == 0), PENALTIES["Device Available But No Recent Vitals"], 0.0)
    penalty_pts += np.where(gap_days.isna(), PENALTIES["No Completed Call"], 0.0)
    penalty_pts += np.where(gap_days.notna() & (gap_days > 90), PENALTIES["Call Gap More Than 90 Days"], 0.0)
    penalty_pts += np.where(gap_days.notna() & (gap_days > 60) & (gap_days <= 90), PENALTIES["Call Gap 61-90 Days"], 0.0)
    penalty_pts += np.where(gap_days.notna() & (gap_days > 30) & (gap_days <= 60), PENALTIES["Call Gap 31-60 Days"], 0.0)
    manual_score = (current_manual - penalty_pts).clip(0, 100)

    model = train_logistic(history, hist_comp)
    logistic_score = predict(current, curr_comp, model)
    final_score = (manual_score * MANUAL_FINAL_WEIGHT + logistic_score * LOGISTIC_FINAL_WEIGHT).clip(0, 100)

    id_col = next((c for c in ["Patient ID", "Patient Id", "Identifier", "Patient Identifier"] if c in current.columns), None)
    call_df = current.copy()
    if id_col:
        call_df["_Patient_Key"] = normalize_identifier(call_df[id_col])
    call_df["Manual Score"] = manual_score.round(1)
    call_df["Logistic Model Score"] = logistic_score.round(1)
    call_df["Final Renewal Score"] = final_score.round(1)
    call_df["Final Renewal Possibility"] = possibility(final_score)
    call_df["Logistic Chance Label"] = logistic_label(logistic_score)
    call_df["Caller"] = call_df.apply(decide_caller, axis=1)
    call_df["Caller Role"] = call_df.apply(determine_caller_role, axis=1)
    call_df["Positive Highlights"] = call_df.apply(positive_highlights, axis=1)
    call_df["Risk Factors"] = call_df.apply(risk_factors, axis=1)
    call_df["Suggested Pitch"] = call_df.apply(suggested_pitch, axis=1)

    esc_df = pd.concat(
        [prepare_escalation(whats, "WhatsApp"), prepare_escalation(mail, "Email")],
        ignore_index=True,
    )
    if "_Patient_Key" in call_df.columns and not esc_df.empty:
        esc_keys = set(esc_df["_Patient_Key"])
        call_df["Escalation Status"] = np.where(call_df["_Patient_Key"].isin(esc_keys), "Escalation", "Normal")
    else:
        call_df["Escalation Status"] = "Normal"

    reason_df = prepare_reason_data(reason)
    if "_Patient_Key" in call_df.columns and not reason_df.empty:
        call_df = call_df.merge(reason_df, on="_Patient_Key", how="left")
    else:
        call_df["Reason"] = ""
        call_df["Reason Comment"] = ""
        call_df["Reason Completion Time"] = pd.NaT

    completed_calls = (
        numeric_series(call_df, "LCM Calls Completed", 0).fillna(0)
        + numeric_series(call_df, "HC Calls Completed", 0).fillna(0)
        + numeric_series(call_df, "CT Calls Completed", 0).fillna(0)
    )
    has_reason = call_df["Reason"].fillna("").astype(str).str.strip().ne("")
    pending_df = call_df[(completed_calls >= 3) & (~has_reason)].copy()
    received_df = call_df[has_reason].copy()

    renewed_count = int(renewal_outcome(history).eq(1).sum())

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Calling List", len(call_df))
    m2.metric("High Possibility", int(call_df["Final Renewal Possibility"].eq("HIGH").sum()))
    m3.metric("Escalations", int(call_df["Escalation Status"].eq("Escalation").sum()))
    m4.metric("Reason Pending (≥3 Calls)", len(pending_df))

    st.divider()
    st.subheader("🔍 Filters & Caller View")
    f1, f2, f3, f4 = st.columns(4)
    callers = ["All"] + sorted(call_df["Caller"].dropna().astype(str).unique().tolist())
    roles = ["All"] + sorted(call_df["Caller Role"].dropna().astype(str).unique().tolist())
    possibilities = ["All", "HIGH", "MODERATE", "LOW"]
    esc_options = ["All", "Escalation", "Normal"]

    sel_caller = f1.selectbox("Caller", callers)
    sel_role = f2.selectbox("Caller Role", roles)
    sel_poss = f3.selectbox("Renewal Possibility", possibilities)
    sel_esc = f4.selectbox("Escalation Status", esc_options)

    search_q = st.text_input("Search by Patient Name or ID", placeholder="Type name or identifier...")

    filtered_df = call_df.copy()
    if sel_caller != "All":
        filtered_df = filtered_df[filtered_df["Caller"].astype(str) == sel_caller]
    if sel_role != "All":
        filtered_df = filtered_df[filtered_df["Caller Role"].astype(str) == sel_role]
    if sel_poss != "All":
        filtered_df = filtered_df[filtered_df["Final Renewal Possibility"].astype(str) == sel_poss]
    if sel_esc != "All":
        filtered_df = filtered_df[filtered_df["Escalation Status"].astype(str) == sel_esc]
    if search_q.strip():
        q = search_q.strip().lower()
        name_s = filtered_df["Patient Name"].astype(str).str.lower() if "Patient Name" in filtered_df.columns else pd.Series("", index=filtered_df.index)
        id_s = filtered_df[id_col].astype(str).str.lower() if id_col else pd.Series("", index=filtered_df.index)
        filtered_df = filtered_df[name_s.str.contains(q, na=False) | id_s.str.contains(q, na=False)]

    st.subheader(f"Final Renewal Call List ({len(filtered_df)} patients)")
    st.dataframe(filtered_df.drop(columns=["_Patient_Key"], errors="ignore"), use_container_width=True)

    st.divider()
    st.subheader("🩺 Patient Drill-Down")
    if not filtered_df.empty:
        def format_patient_option(idx):
            row = filtered_df.loc[idx]
            p_name = str(row.get("Patient Name", "") or "Unknown").strip()
            p_id = str(row.get(id_col, idx) if id_col else idx).strip()
            score = row.get("Final Renewal Score", "")
            return f"{p_id} — {p_name} (Score: {score})"

        selected_idx = st.selectbox("Select a patient to inspect details", options=filtered_df.index.tolist(), format_func=format_patient_option)
        p_row = filtered_df.loc[selected_idx]

        d1, d2, d3, d4 = st.columns(4)
        d1.metric("Manual Score", p_row.get("Manual Score", "-"))
        d2.metric("Logistic Model Score", p_row.get("Logistic Model Score", "-"))
        d3.metric("Final Renewal Score", p_row.get("Final Renewal Score", "-"))
        d4.metric("Caller", f"{p_row.get('Caller', 'Unassigned')} ({p_row.get('Caller Role', '-')})")

        st.markdown(f"**Possibility:** {p_row.get('Final Renewal Possibility', '-')} | **Logistic Chance:** {p_row.get('Logistic Chance Label', '-')} | **Escalation Status:** {p_row.get('Escalation Status', '-')}")
        st.markdown(f"**Positive Highlights:** {p_row.get('Positive Highlights', '-')}")
        st.markdown(f"**Risk Factors:** {p_row.get('Risk Factors', '-')}")
        st.info(f"**Suggested Pitch:** {p_row.get('Suggested Pitch', '-')}")

        if "_Patient_Key" in p_row and not esc_df.empty:
            p_esc = esc_df[esc_df["_Patient_Key"] == p_row["_Patient_Key"]]
            if not p_esc.empty:
                st.warning("Escalation Records for this Patient:")
                st.dataframe(p_esc.drop(columns=["_Patient_Key"], errors="ignore"), use_container_width=True)

        with st.expander("View All Raw Fields for Selected Patient"):
            st.json({k: (None if pd.isna(v) else str(v)) for k, v in p_row.drop(labels=["_Patient_Key"], errors="ignore").items()})
    else:
        st.info("No patients match the selected filters.")

    st.divider()
    excel_bytes_out = renewal_report_excel(call_df, pending_df, received_df)
    st.download_button(
        "⬇️ Download Renewal Report (.xlsx)",
        data=excel_bytes_out,
        file_name=f"Lyfe_Renewal_Report_{TODAY}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

    with st.expander("📧 Send Report via SendGrid"):
        api_key = st.text_input("SendGrid API Key", type="password")
        sender = st.text_input("From Email", value=DEFAULT_REPORT_SENDER)
        to_raw = st.text_input("To Emails (comma separated)", value=DEFAULT_REPORT_RECIPIENTS)
        bcc_raw = st.text_input("BCC Emails (comma separated)", value=DEFAULT_REPORT_BCC)
        subject = st.text_input("Subject", value=f"Lyfe Renewal Intelligence Report - {TODAY}")
        if st.button("Send Email"):
            to_list = parse_email_list(to_raw)
            bcc_list = parse_email_list(bcc_raw)
            if not api_key.strip():
                st.error("Please enter SendGrid API Key.")
            elif not is_valid_email(sender) or not to_list:
                st.error("Please enter a valid sender and at least one valid recipient email.")
            else:
                html_body = build_report_html(call_df, pending_df, received_df, renewed_count)
                msg = create_report_email(
                    sender.strip(),
                    to_list,
                    [],
                    bcc_list,
                    subject.strip(),
                    html_body,
                    excel_bytes_out,
                    f"Lyfe_Renewal_Report_{TODAY}.xlsx",
                )
                server = connect_sendgrid(api_key)
                try:
                    server.send_message(msg)
                finally:
                    server.quit()
                st.success("Report email sent successfully.")

except Exception as e:
    st.error(f"Error processing Excel file: {e}")
