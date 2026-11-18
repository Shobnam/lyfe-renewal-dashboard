import streamlit as st
import pandas as pd
import numpy as np


# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="Lyfe Renewal Possibility App",
    page_icon="🩺",
    layout="wide"
)

st.title("🩺 Lyfe Renewal Possibility App")

st.caption(
    "Patient engagement, CT/Psychotherapy effort, device usage, "
    "clinical improvement, NPS and call recency ke basis par "
    "renewal possibility calculate hoti hai."
)


# =========================================================
# SIDEBAR WEIGHTAGE
# =========================================================
st.sidebar.header("⚖️ Renewal Score Weightage")

st.sidebar.caption(
    "Har factor ka maximum contribution adjust kar sakte hain."
)

w_lcm = st.sidebar.slider(
    "LCM Engagement",
    min_value=0,
    max_value=30,
    value=10
)

w_hc = st.sidebar.slider(
    "HC Engagement",
    min_value=0,
    max_value=30,
    value=10
)

w_ct = st.sidebar.slider(
    "CT Engagement",
    min_value=0,
    max_value=30,
    value=20
)

w_psycho = st.sidebar.slider(
    "Psychotherapist Engagement",
    min_value=0,
    max_value=30,
    value=15
)

w_device = st.sidebar.slider(
    "Device + Vital Usage",
    min_value=0,
    max_value=30,
    value=15
)

w_clinical = st.sidebar.slider(
    "Clinical Improvement",
    min_value=0,
    max_value=30,
    value=20
)

w_nps = st.sidebar.slider(
    "NPS",
    min_value=0,
    max_value=30,
    value=10
)


total_configured_weight = (
    w_lcm
    + w_hc
    + w_ct
    + w_psycho
    + w_device
    + w_clinical
    + w_nps
)


st.sidebar.metric(
    "Configured Weight Total",
    total_configured_weight
)

if total_configured_weight == 100:

    st.sidebar.success(
        "✅ Total weightage 100 hai."
    )

else:

    st.sidebar.info(
        "Score automatically normalize hoga, "
        "lekin recommended total 100 hai."
    )


# =========================================================
# PENALTIES
# =========================================================
st.sidebar.divider()

st.sidebar.header("🚨 Risk Penalties")

silent_device_penalty = st.sidebar.slider(
    "Device available but no vitals",
    min_value=0,
    max_value=30,
    value=15
)

call_gap_31_60_penalty = st.sidebar.slider(
    "Last call gap: 31–60 days",
    min_value=0,
    max_value=10,
    value=2
)

call_gap_61_90_penalty = st.sidebar.slider(
    "Last call gap: 61–90 days",
    min_value=0,
    max_value=15,
    value=5
)

call_gap_90_plus_penalty = st.sidebar.slider(
    "Last call gap: More than 90 days",
    min_value=0,
    max_value=20,
    value=8
)

no_completed_call_penalty = st.sidebar.slider(
    "No completed call",
    min_value=0,
    max_value=20,
    value=10
)


# =========================================================
# FILE UPLOAD
# =========================================================
uploaded_file = st.file_uploader(
    "📂 Upload Patient File",
    type=["csv", "xlsx"],
    help="CSV ya Excel file upload karein."
)


if uploaded_file is None:

    st.info(
        "👆 Renewal patient CSV ya Excel file upload karein."
    )

    st.stop()


# =========================================================
# READ CSV / EXCEL
# =========================================================
try:

    if uploaded_file.name.lower().endswith(".csv"):

        df = pd.read_csv(
            uploaded_file
        )

        selected_sheet = None


    elif uploaded_file.name.lower().endswith(".xlsx"):

        excel_file = pd.ExcelFile(
            uploaded_file,
            engine="openpyxl"
        )

        sheet_names = excel_file.sheet_names

        detected_sheet = None


        # Subscription End Date wali sheet detect karo
        for sheet_name in sheet_names:

            try:

                sample_df = pd.read_excel(
                    uploaded_file,
                    sheet_name=sheet_name,
                    engine="openpyxl",
                    nrows=5
                )

                sample_df.columns = (
                    sample_df.columns
                    .astype(str)
                    .str.strip()
                )

                if "Subscription End Date" in sample_df.columns:

                    detected_sheet = sheet_name
                    break

            except Exception:

                continue


        if detected_sheet in sheet_names:

            default_sheet_index = sheet_names.index(
                detected_sheet
            )

        else:

            default_sheet_index = 0


        selected_sheet = st.selectbox(
            "📄 Select Excel Sheet",
            sheet_names,
            index=default_sheet_index
        )


        df = pd.read_excel(
            uploaded_file,
            sheet_name=selected_sheet,
            engine="openpyxl"
        )


    else:

        st.error(
            "Only CSV aur Excel .xlsx file supported hai."
        )

        st.stop()


except Exception as error:

    st.error(
        "File read nahi ho paayi."
    )

    st.exception(error)

    st.stop()


# =========================================================
# CLEAN COLUMN NAMES
# =========================================================
df.columns = (
    df.columns
    .astype(str)
    .str.strip()
)


st.success(
    f"✅ File loaded: {uploaded_file.name}"
)

if selected_sheet is not None:

    st.caption(
        f"Selected Excel sheet: {selected_sheet}"
    )


# =========================================================
# REQUIRED DATE COLUMN
# =========================================================
if "Subscription End Date" not in df.columns:

    st.error(
        "`Subscription End Date` column file mein nahi mila."
    )

    st.write("Available columns:")

    st.write(
        df.columns.tolist()
    )

    st.stop()


# =========================================================
# DATE CONVERSION
# =========================================================
date_columns = [
    "Subscription Start Date",
    "Subscription End Date",
    "Last Completed Call Date",
    "Last Vital Date",
    "Initial Active Weight Date",
    "Latest Active Weight Date",
    "Baseline Record Date",
    "Current SBP Date",
    "Current LDL Date",
    "Current HbA1c Date",
    "Latest NPS Date"
]


for column in date_columns:

    if column in df.columns:

        df[column] = pd.to_datetime(
            df[column],
            errors="coerce"
        )


df = df[
    df["Subscription End Date"].notna()
].copy()


if df.empty:

    st.warning(
        "Valid Subscription End Date wala koi record nahi mila."
    )

    st.stop()


# =========================================================
# RENEWAL MONTH
# =========================================================
df["Renewal Month"] = (
    df["Subscription End Date"]
    .dt.strftime("%b %Y")
)

df["Renewal Month Sort"] = (
    df["Subscription End Date"]
    .dt.to_period("M")
    .dt.to_timestamp()
)


month_table = (
    df[
        [
            "Renewal Month",
            "Renewal Month Sort"
        ]
    ]
    .drop_duplicates()
    .sort_values(
        "Renewal Month Sort"
    )
)


month_options = (
    month_table["Renewal Month"]
    .tolist()
)


st.divider()

st.subheader("📅 Select Renewal Month")


selected_month = st.selectbox(
    "Which month's renewal patients do you want to see?",
    month_options
)


filtered_df = df[
    df["Renewal Month"] == selected_month
].copy()


# =========================================================
# HELPER FUNCTIONS
# =========================================================
def numeric_value(row, column_name, default=np.nan):

    if column_name not in row.index:

        return default

    value = pd.to_numeric(
        row[column_name],
        errors="coerce"
    )

    if pd.isna(value):

        return default

    return float(value)


def text_value(row, column_name):

    if column_name not in row.index:

        return ""

    if pd.isna(row[column_name]):

        return ""

    return str(
        row[column_name]
    ).strip().upper()


def yes_value(row, column_name):

    value = text_value(
        row,
        column_name
    )

    return value in [
        "YES",
        "Y",
        "TRUE",
        "1"
    ]


# =========================================================
# SERVICE ENGAGEMENT SCORE
#
# Rate + completed count effort
# Output: 0 to 100
# =========================================================
def service_engagement_score(
    row,
    scheduled_column,
    completed_column,
    completion_column,
    strong_completed_target
):

    scheduled = numeric_value(
        row,
        scheduled_column,
        0
    )

    completed = numeric_value(
        row,
        completed_column,
        0
    )

    completion_pct = numeric_value(
        row,
        completion_column,
        np.nan
    )


    # Service scheduled hi nahi hua
    if scheduled <= 0:

        return np.nan


    # Completion percentage missing ho to calculate karo
    if pd.isna(completion_pct):

        completion_pct = (
            completed / scheduled
        ) * 100


    completion_pct = np.clip(
        completion_pct,
        0,
        100
    )


    # Actual completed sessions effort
    effort_pct = min(
        completed / strong_completed_target,
        1
    ) * 100


    # 70% completion consistency
    # 30% completed session effort
    final_service_score = (
        completion_pct * 0.70
        +
        effort_pct * 0.30
    )


    return round(
        final_service_score,
        2
    )


# =========================================================
# DEVICE + VITAL SCORE
# =========================================================
def device_vital_score(row):

    expected_devices = numeric_value(
        row,
        "Vital Devices Available",
        np.nan
    )


    # Fallback: individual device flags
    if pd.isna(expected_devices):

        expected_devices = (
            int(
                yes_value(
                    row,
                    "BP Monitor Available"
                )
            )
            +
            int(
                yes_value(
                    row,
                    "Weight Scale Available"
                )
            )
            +
            int(
                yes_value(
                    row,
                    "Glucometer Available"
                )
            )
            +
            int(
                yes_value(
                    row,
                    "Oximeter Available"
                )
            )
        )


    # Second fallback
    if expected_devices <= 0:

        active_device_count = numeric_value(
            row,
            "Active Device Count",
            0
        )

        expected_devices = active_device_count


    # No relevant device means not applicable
    if expected_devices <= 0:

        return np.nan


    devices_with_vitals = numeric_value(
        row,
        "Devices With Recent Vitals",
        np.nan
    )


    coverage_pct = numeric_value(
        row,
        "Device Vital Coverage %",
        np.nan
    )


    if pd.isna(coverage_pct):

        if not pd.isna(devices_with_vitals):

            coverage_pct = (
                devices_with_vitals
                / expected_devices
            ) * 100

        else:

            # Individual device usage check
            active_used_devices = 0

            if (
                yes_value(
                    row,
                    "BP Monitor Available"
                )
                and
                numeric_value(
                    row,
                    "BP Readings - Last 30 Days",
                    0
                ) > 0
            ):

                active_used_devices += 1


            if (
                yes_value(
                    row,
                    "Weight Scale Available"
                )
                and
                numeric_value(
                    row,
                    "Weight Readings - Last 30 Days",
                    0
                ) > 0
            ):

                active_used_devices += 1


            if (
                yes_value(
                    row,
                    "Glucometer Available"
                )
                and
                numeric_value(
                    row,
                    "Glucose Readings - Last 30 Days",
                    0
                ) > 0
            ):

                active_used_devices += 1


            if (
                yes_value(
                    row,
                    "Oximeter Available"
                )
                and
                numeric_value(
                    row,
                    "SpO2 Readings - Last 30 Days",
                    0
                ) > 0
            ):

                active_used_devices += 1


            coverage_pct = (
                active_used_devices
                / expected_devices
            ) * 100


    coverage_pct = np.clip(
        coverage_pct,
        0,
        100
    )


    total_vitals = numeric_value(
        row,
        "Total Relevant Vitals - Last 30 Days",
        0
    )


    # Expected 4 readings per active device per month
    expected_reading_target = max(
        expected_devices * 4,
        1
    )


    reading_activity_pct = min(
        total_vitals / expected_reading_target,
        1
    ) * 100


    # Coverage is more important than total volume
    score = (
        coverage_pct * 0.80
        +
        reading_activity_pct * 0.20
    )


    return round(
        score,
        2
    )


# =========================================================
# CLINICAL IMPROVEMENT SCORE
# =========================================================
def clinical_improvement_score(row):

    metric_scores = []


    # Weight
    weight_trend = text_value(
        row,
        "Weight Trend"
    )

    if weight_trend:

        if "REDUCED" in weight_trend:

            metric_scores.append(100)

        elif "NO CHANGE" in weight_trend:

            metric_scores.append(50)

        elif "INCREASED" in weight_trend:

            metric_scores.append(0)


    # SBP
    sbp_trend = text_value(
        row,
        "SBP Trend"
    )

    if sbp_trend:

        if "REDUCED" in sbp_trend:

            metric_scores.append(100)

        elif "NO CHANGE" in sbp_trend:

            metric_scores.append(50)

        elif "INCREASED" in sbp_trend:

            metric_scores.append(0)


    # LDL
    ldl_trend = text_value(
        row,
        "LDL Trend"
    )

    if ldl_trend:

        if "REDUCED" in ldl_trend:

            metric_scores.append(100)

        elif "NO CHANGE" in ldl_trend:

            metric_scores.append(50)

        elif "INCREASED" in ldl_trend:

            metric_scores.append(0)


    # HbA1c
    hba1c_trend = text_value(
        row,
        "HbA1c Trend"
    )

    if hba1c_trend:

        if "REDUCED" in hba1c_trend:

            metric_scores.append(100)

        elif "NO CHANGE" in hba1c_trend:

            metric_scores.append(50)

        elif "INCREASED" in hba1c_trend:

            metric_scores.append(0)


    # Missing clinical data is neutral/not applicable
    if len(metric_scores) == 0:

        return np.nan


    return round(
        np.mean(metric_scores),
        2
    )


# =========================================================
# NPS SCORE
# =========================================================
def nps_component_score(row):

    latest_nps = numeric_value(
        row,
        "Latest NPS Score",
        np.nan
    )


    if pd.isna(latest_nps):

        return np.nan


    latest_nps = np.clip(
        latest_nps,
        0,
        10
    )


    return round(
        latest_nps * 10,
        2
    )


# =========================================================
# SILENT DEVICE FLAG
# =========================================================
def silent_device_flag(row):

    active_devices = numeric_value(
        row,
        "Active Device Count",
        0
    )


    expected_devices = numeric_value(
        row,
        "Vital Devices Available",
        active_devices
    )


    recent_vitals = numeric_value(
        row,
        "Total Relevant Vitals - Last 30 Days",
        0
    )


    return (
        expected_devices > 0
        and
        recent_vitals == 0
    )


# =========================================================
# CALL RECENCY PENALTY
# =========================================================
today = pd.Timestamp.today().normalize()


def call_recency_details(row):

    if "Last Completed Call Date" not in row.index:

        return np.nan, 0, "Column Not Available"


    last_call_date = pd.to_datetime(
        row["Last Completed Call Date"],
        errors="coerce"
    )


    if pd.isna(last_call_date):

        return (
            np.nan,
            no_completed_call_penalty,
            "🚨 No Completed Call"
        )


    days_since_call = (
        today - last_call_date.normalize()
    ).days


    # Future dates ko 0 treat karo
    days_since_call = max(
        days_since_call,
        0
    )


    if days_since_call <= 30:

        return (
            days_since_call,
            0,
            "🟢 Active: 0–30 Days"
        )


    elif days_since_call <= 60:

        return (
            days_since_call,
            call_gap_31_60_penalty,
            "🟡 Gap: 31–60 Days"
        )


    elif days_since_call <= 90:

        return (
            days_since_call,
            call_gap_61_90_penalty,
            "🟠 Gap: 61–90 Days"
        )


    else:

        return (
            days_since_call,
            call_gap_90_plus_penalty,
            "🔴 Gap: More Than 90 Days"
        )


# =========================================================
# CALCULATE PATIENT SCORE
# =========================================================
def calculate_patient_score(row):

    component_values = {
        "LCM": service_engagement_score(
            row,
            "LCM Calls Scheduled",
            "LCM Calls Completed",
            "LCM Completion %",
            strong_completed_target=6
        ),

        "HC": service_engagement_score(
            row,
            "HC Calls Scheduled",
            "HC Calls Completed",
            "HC Completion %",
            strong_completed_target=6
        ),

        "CT": service_engagement_score(
            row,
            "CT Calls Scheduled",
            "CT Calls Completed",
            "CT Completion %",
            strong_completed_target=6
        ),

        "Psychotherapist": service_engagement_score(
            row,
            "Psychotherapist Sessions Scheduled",
            "Psychotherapist Sessions Completed",
            "Psychotherapist Completion %",
            strong_completed_target=4
        ),

        "Device": device_vital_score(
            row
        ),

        "Clinical": clinical_improvement_score(
            row
        ),

        "NPS": nps_component_score(
            row
        )
    }


    component_weights = {
        "LCM": w_lcm,
        "HC": w_hc,
        "CT": w_ct,
        "Psychotherapist": w_psycho,
        "Device": w_device,
        "Clinical": w_clinical,
        "NPS": w_nps
    }


    weighted_points = 0
    applicable_weight = 0


    for component_name in component_values:

        value = component_values[
            component_name
        ]

        weight = component_weights[
            component_name
        ]


        # Missing/not applicable factor denominator mein nahi jayega
        if not pd.isna(value) and weight > 0:

            weighted_points += (
                value / 100
            ) * weight

            applicable_weight += weight


    if applicable_weight > 0:

        base_score = (
            weighted_points
            / applicable_weight
        ) * 100

    else:

        base_score = 0


    # Device penalty
    is_silent_device = silent_device_flag(
        row
    )


    device_penalty = (
        silent_device_penalty
        if is_silent_device
        else 0
    )


    # Call gap penalty
    (
        days_since_call,
        call_penalty,
        call_gap_status
    ) = call_recency_details(
        row
    )


    total_penalty = (
        device_penalty
        +
        call_penalty
    )


    final_score = np.clip(
        base_score - total_penalty,
        0,
        100
    )


    if final_score >= 75:

        possibility = "🟢 High Possibility"

    elif final_score >= 50:

        possibility = "🟡 Moderate Possibility"

    else:

        possibility = "🔴 Low Possibility"


    return pd.Series({
        "LCM Score": component_values["LCM"],
        "HC Score": component_values["HC"],
        "CT Score": component_values["CT"],
        "Psychotherapist Score": component_values["Psychotherapist"],
        "Device Vital Score": component_values["Device"],
        "Clinical Improvement Score": component_values["Clinical"],
        "NPS Score Component": component_values["NPS"],

        "Applicable Weight": applicable_weight,
        "Base Renewal Score": round(base_score, 1),

        "Silent Device Flag": (
            "🚨 Device Active But No Recent Vitals"
            if is_silent_device
            else "OK"
        ),

        "Device Penalty": device_penalty,
        "Days Since Last Completed Call": days_since_call,
        "Call Gap Status": call_gap_status,
        "Call Gap Penalty": call_penalty,
        "Total Penalty": total_penalty,

        "Renewal Score": round(final_score, 1),
        "Renewal Possibility": possibility
    })


# =========================================================
# APPLY SCORE
# =========================================================
score_df = filtered_df.apply(
    calculate_patient_score,
    axis=1
)


filtered_df = pd.concat(
    [
        filtered_df.reset_index(drop=True),
        score_df.reset_index(drop=True)
    ],
    axis=1
)


# =========================================================
# SUMMARY METRICS
# =========================================================
high_count = (
    filtered_df["Renewal Possibility"]
    == "🟢 High Possibility"
).sum()


moderate_count = (
    filtered_df["Renewal Possibility"]
    == "🟡 Moderate Possibility"
).sum()


low_count = (
    filtered_df["Renewal Possibility"]
    == "🔴 Low Possibility"
).sum()


silent_device_count = (
    filtered_df["Silent Device Flag"]
    != "OK"
).sum()


call_gap_count = (
    filtered_df[
        "Days Since Last Completed Call"
    ] > 60
).sum()


st.success(
    f"✅ {len(filtered_df)} patients found whose plan ends in "
    f"{selected_month}"
)


st.subheader(
    f"📊 {selected_month} Renewal Summary"
)


metric1, metric2, metric3, metric4, metric5, metric6 = (
    st.columns(6)
)


metric1.metric(
    "Total Patients",
    len(filtered_df)
)

metric2.metric(
    "🟢 High",
    int(high_count)
)

metric3.metric(
    "🟡 Moderate",
    int(moderate_count)
)

metric4.metric(
    "🔴 Low",
    int(low_count)
)

metric5.metric(
    "🚨 Silent Devices",
    int(silent_device_count)
)

metric6.metric(
    "📞 Call Gap >60d",
    int(call_gap_count)
)


# =========================================================
# FILTERS
# =========================================================
st.divider()

st.subheader("🔍 Patient Filters")


filter_col1, filter_col2, filter_col3 = st.columns(3)


selected_possibility = filter_col1.multiselect(
    "Renewal Possibility",
    [
        "🟢 High Possibility",
        "🟡 Moderate Possibility",
        "🔴 Low Possibility"
    ],
    default=[
        "🟢 High Possibility",
        "🟡 Moderate Possibility",
        "🔴 Low Possibility"
    ]
)


show_silent_devices = filter_col2.checkbox(
    "Only Silent Device Patients"
)


show_call_gap = filter_col3.checkbox(
    "Only Call Gap More Than 60 Days"
)


view_df = filtered_df[
    filtered_df["Renewal Possibility"]
    .isin(selected_possibility)
].copy()


if show_silent_devices:

    view_df = view_df[
        view_df["Silent Device Flag"] != "OK"
    ].copy()


if show_call_gap:

    view_df = view_df[
        view_df[
            "Days Since Last Completed Call"
        ] > 60
    ].copy()


# =========================================================
# RENEWAL POSSIBILITY TABLE
# =========================================================
st.divider()

st.subheader(
    f"🎯 {selected_month} Renewal Possibility"
)


display_columns = [
    "Identifier",
    "Patient Name",
    "Subscription End Date",
    "Latest Engagement Status",
    "HC Name",
    "LCM Name",
    "CT Physio Name",

    "LCM Calls Scheduled",
    "LCM Calls Completed",
    "LCM Completion %",

    "HC Calls Scheduled",
    "HC Calls Completed",
    "HC Completion %",

    "CT Calls Scheduled",
    "CT Calls Completed",
    "CT Completion %",

    "Psychotherapist Sessions Scheduled",
    "Psychotherapist Sessions Completed",
    "Psychotherapist Completion %",

    "Device Vital Coverage %",
    "Total Relevant Vitals - Last 30 Days",
    "Silent Device Flag",

    "Weight Trend",
    "SBP Trend",
    "LDL Trend",
    "HbA1c Trend",

    "Latest NPS Score",
    "Last Completed Call Date",
    "Days Since Last Completed Call",
    "Call Gap Status",

    "Base Renewal Score",
    "Total Penalty",
    "Renewal Score",
    "Renewal Possibility"
]


available_display_columns = [
    column
    for column in display_columns
    if column in view_df.columns
]


table_df = view_df[
    available_display_columns
].sort_values(
    "Renewal Score",
    ascending=False
)


st.dataframe(
    table_df,
    use_container_width=True,
    hide_index=True
)


# =========================================================
# SCORE BREAKDOWN
# =========================================================
st.divider()

st.subheader("🧮 Patient Score Breakdown")


breakdown_columns = [
    "Identifier",
    "Patient Name",
    "LCM Score",
    "HC Score",
    "CT Score",
    "Psychotherapist Score",
    "Device Vital Score",
    "Clinical Improvement Score",
    "NPS Score Component",
    "Applicable Weight",
    "Base Renewal Score",
    "Device Penalty",
    "Call Gap Penalty",
    "Total Penalty",
    "Renewal Score",
    "Renewal Possibility"
]


available_breakdown_columns = [
    column
    for column in breakdown_columns
    if column in view_df.columns
]


st.dataframe(
    view_df[
        available_breakdown_columns
    ].sort_values(
        "Renewal Score",
        ascending=False
    ),
    use_container_width=True,
    hide_index=True
)


# =========================================================
# DOWNLOAD
# =========================================================
st.divider()


download_df = view_df.drop(
    columns=[
        "Renewal Month Sort"
    ],
    errors="ignore"
)


csv_data = download_df.to_csv(
    index=False
).encode("utf-8")


safe_month = (
    selected_month
    .replace(" ", "_")
    .lower()
)


st.download_button(
    label=f"📥 Download {selected_month} Scored Patients",
    data=csv_data,
    file_name=f"{safe_month}_renewal_possibility.csv",
    mime="text/csv"
)


# =========================================================
# MODEL NOTE
# =========================================================
st.info(
    "ℹ️ Ye business-rule based renewal possibility score hai. "
    "Ye actual confirmed renewal probability nahi hai. "
    "Historical renewed vs non-renewed data milne ke baad "
    "weights aur cut-offs validate kiye ja sakte hain."
)
