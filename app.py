import streamlit as st
import pandas as pd


# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="Lyfe Renewal App",
    page_icon="🩺",
    layout="wide"
)

st.title("🩺 Lyfe Renewal App")

st.write(
    "Patient file upload karein aur renewal month select karke "
    "us month mein plan end hone wale patients dekhein."
)


# =========================================================
# 1. FILE UPLOAD
# =========================================================
uploaded_file = st.file_uploader(
    "📂 Upload Patient File (CSV or Excel)",
    type=["csv", "xlsx"]
)


if uploaded_file is None:
    st.info("👆 CSV ya Excel patient file upload karein.")
    st.stop()


# =========================================================
# 2. READ CSV / EXCEL
# =========================================================
try:

    if uploaded_file.name.lower().endswith(".csv"):

        df = pd.read_csv(
            uploaded_file
        )

    elif uploaded_file.name.lower().endswith(".xlsx"):

        df = pd.read_excel(
            uploaded_file,
            engine="openpyxl"
        )

    else:

        st.error("Only CSV aur Excel (.xlsx) files supported hain.")
        st.stop()


except Exception as e:

    st.error("File read nahi ho paayi.")
    st.exception(e)
    st.stop()


st.success(
    f"✅ File uploaded successfully: {uploaded_file.name}"
)


# =========================================================
# 3. CLEAN COLUMN NAMES
# =========================================================
df.columns = df.columns.astype(str).str.strip()


# =========================================================
# 4. CHECK SUBSCRIPTION END DATE
# =========================================================
required_date_column = "Subscription End Date"

if required_date_column not in df.columns:

    st.error(
        "❌ File mein `Subscription End Date` column nahi mila."
    )

    st.write("File mein available columns:")

    st.write(
        df.columns.tolist()
    )

    st.stop()


# =========================================================
# 5. CONVERT SUBSCRIPTION END DATE
# =========================================================
df["Subscription End Date"] = pd.to_datetime(
    df["Subscription End Date"],
    errors="coerce",
    dayfirst=False
)


# Invalid / blank end dates remove
df = df[
    df["Subscription End Date"].notna()
].copy()


if df.empty:

    st.warning(
        "File mein valid Subscription End Date available nahi hai."
    )

    st.stop()


# =========================================================
# 6. CREATE RENEWAL MONTH
# Example: Jan 2026
# =========================================================
df["Renewal Month"] = (
    df["Subscription End Date"]
    .dt.strftime("%b %Y")
)


# =========================================================
# 7. CREATE MONTH LIST FROM ACTUAL FILE DATA
# =========================================================
month_table = (
    df[
        [
            "Subscription End Date",
            "Renewal Month"
        ]
    ]
    .assign(
        Month_Start=lambda x:
        x["Subscription End Date"]
        .dt.to_period("M")
        .dt.to_timestamp()
    )
    [
        [
            "Renewal Month",
            "Month_Start"
        ]
    ]
    .drop_duplicates()
    .sort_values(
        "Month_Start"
    )
)


month_options = (
    month_table["Renewal Month"]
    .tolist()
)


if not month_options:

    st.warning(
        "Renewal month available nahi hai."
    )

    st.stop()


# =========================================================
# 8. SELECT RENEWAL MONTH
# =========================================================
st.divider()

st.subheader("📅 Select Renewal Month")


selected_month = st.selectbox(
    "Which month's renewal patients do you want to see?",
    month_options
)


# =========================================================
# 9. FILTER SELECTED MONTH
# =========================================================
filtered_df = df[
    df["Renewal Month"] == selected_month
].copy()


# =========================================================
# 10. NPS CLEANING
# =========================================================
if "Latest NPS Score" in filtered_df.columns:

    filtered_df["Latest NPS Score"] = pd.to_numeric(
        filtered_df["Latest NPS Score"],
        errors="coerce"
    )


    # -----------------------------------------------------
    # NPS CATEGORY
    # -----------------------------------------------------
    def get_nps_category(score):

        if pd.isna(score):
            return "No NPS"

        if score >= 9:
            return "🟢 Promoter"

        elif score >= 7:
            return "🟡 Passive"

        else:
            return "🔴 Detractor"


    filtered_df["NPS Category"] = (
        filtered_df["Latest NPS Score"]
        .apply(get_nps_category)
    )


# =========================================================
# 11. SELECTED MONTH SUMMARY
# =========================================================
st.success(
    f"✅ {len(filtered_df)} patients found whose plan ends in "
    f"{selected_month}"
)


st.subheader(
    f"📊 {selected_month} Renewal Summary"
)


col1, col2, col3, col4 = st.columns(4)


# Total patients
col1.metric(
    "Total Renewal Patients",
    len(filtered_df)
)


# =========================================================
# NPS KPI
# =========================================================
if "Latest NPS Score" in filtered_df.columns:

    valid_nps = filtered_df[
        filtered_df["Latest NPS Score"].notna()
    ].copy()


    # Average
    avg_nps = (
        valid_nps["Latest NPS Score"].mean()
        if not valid_nps.empty
        else None
    )


    col2.metric(
        "Average NPS",
        round(avg_nps, 1)
        if pd.notna(avg_nps)
        else "-"
    )


    # Promoters 9-10
    promoters = (
        valid_nps["Latest NPS Score"] >= 9
    ).sum()


    col3.metric(
        "🟢 Promoters",
        int(promoters)
    )


    # Detractors 0-6
    detractors = (
        valid_nps["Latest NPS Score"] <= 6
    ).sum()


    col4.metric(
        "🔴 Detractors",
        int(detractors)
    )


else:

    col2.metric(
        "Average NPS",
        "-"
    )

    col3.metric(
        "🟢 Promoters",
        "-"
    )

    col4.metric(
        "🔴 Detractors",
        "-"
    )


# =========================================================
# 12. RENEWAL PATIENT TABLE
# =========================================================
st.divider()

st.subheader(
    f"📋 {selected_month} Renewal Patients"
)


patient_columns = [
    "Identifier",
    "Patient Name",
    "Email",
    "Subscription Start Date",
    "Subscription End Date",
    "Latest Engagement Status",
    "HC Name",
    "LCM Name",
    "Latest NPS Score",
    "NPS Category",
    "Latest NPS Comments"
]


available_patient_columns = [
    column
    for column in patient_columns
    if column in filtered_df.columns
]


if available_patient_columns:

    st.dataframe(
        filtered_df[
            available_patient_columns
        ].sort_values(
            "Subscription End Date"
        ),
        use_container_width=True,
        hide_index=True
    )

else:

    st.dataframe(
        filtered_df,
        use_container_width=True,
        hide_index=True
    )


# =========================================================
# 13. NPS DETAILS
# =========================================================
st.divider()

st.subheader("⭐ NPS Details")


if "Latest NPS Score" in filtered_df.columns:

    nps_columns = [
        "Identifier",
        "Patient Name",
        "Latest NPS Score",
        "NPS Category",
        "Latest NPS Date",
        "Latest NPS Comments",
        "Latest NPS Data Source",
        "Average NPS Score",
        "Total NPS Responses",
        "Promoter Responses",
        "Passive Responses",
        "Detractor Responses",
        "Overall Service Feedback",
        "Health Coach Feedback",
        "LCM Feedback",
        "Cardio Therapy Feedback",
        "Meditation Feedback",
        "Monitoring Devices Feedback",
        "Lyfe App Feedback",
        "Alert Management Feedback"
    ]


    available_nps_columns = [
        column
        for column in nps_columns
        if column in filtered_df.columns
    ]


    if available_nps_columns:

        st.dataframe(
            filtered_df[
                available_nps_columns
            ],
            use_container_width=True,
            hide_index=True
        )


else:

    st.warning(
        "`Latest NPS Score` column uploaded file mein nahi mila."
    )


# =========================================================
# 14. DOWNLOAD SELECTED MONTH PATIENTS
# =========================================================
st.divider()


download_csv = (
    filtered_df
    .to_csv(
        index=False
    )
    .encode("utf-8")
)


safe_month_name = (
    selected_month
    .replace(" ", "_")
    .lower()
)


st.download_button(
    label=f"📥 Download {selected_month} Patient List",
    data=download_csv,
    file_name=f"{safe_month_name}_renewal_patients.csv",
    mime="text/csv"
)
