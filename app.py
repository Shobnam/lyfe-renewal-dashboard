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
# FILE UPLOAD
# =========================================================
uploaded_file = st.file_uploader(
    "📂 Upload Patient File (CSV or Excel)",
    type=["csv", "xlsx"]
)


if uploaded_file is None:
    st.info("👆 CSV ya Excel file upload karein.")
    st.stop()


# =========================================================
# READ FILE
# =========================================================
try:

    # -----------------------------------------------------
    # CSV
    # -----------------------------------------------------
    if uploaded_file.name.lower().endswith(".csv"):

        df = pd.read_csv(uploaded_file)

        selected_sheet = None


    # -----------------------------------------------------
    # EXCEL
    # -----------------------------------------------------
    elif uploaded_file.name.lower().endswith(".xlsx"):

        excel_file = pd.ExcelFile(
            uploaded_file,
            engine="openpyxl"
        )

        sheet_names = excel_file.sheet_names


        # -----------------------------------------------
        # TRY TO FIND SHEET WITH SUBSCRIPTION END DATE
        # -----------------------------------------------
        detected_sheet = None

        for sheet in sheet_names:

            try:

                check_df = pd.read_excel(
                    uploaded_file,
                    sheet_name=sheet,
                    engine="openpyxl",
                    nrows=5
                )

                check_df.columns = (
                    check_df.columns
                    .astype(str)
                    .str.strip()
                )

                if "Subscription End Date" in check_df.columns:
                    detected_sheet = sheet
                    break

            except Exception:
                continue


        # -----------------------------------------------
        # SELECT DEFAULT SHEET
        # -----------------------------------------------
        if detected_sheet is not None:
            default_index = sheet_names.index(
                detected_sheet
            )
        else:
            default_index = 0


        selected_sheet = st.selectbox(
            "📄 Select Excel Sheet",
            sheet_names,
            index=default_index
        )


        # -----------------------------------------------
        # READ SELECTED SHEET
        # -----------------------------------------------
        df = pd.read_excel(
            uploaded_file,
            sheet_name=selected_sheet,
            engine="openpyxl"
        )


    else:

        st.error(
            "❌ Only CSV aur Excel (.xlsx) supported hain."
        )

        st.stop()


except Exception as e:

    st.error(
        "❌ File read nahi ho paayi."
    )

    st.exception(e)

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
    f"✅ File uploaded successfully: {uploaded_file.name}"
)


if selected_sheet:
    st.caption(
        f"Excel Sheet: {selected_sheet}"
    )


# =========================================================
# CHECK EMPTY FILE / SHEET
# =========================================================
if len(df.columns) == 0:

    st.error(
        "❌ Selected Excel sheet blank hai. "
        "Please data wali sheet select karein."
    )

    st.stop()


# =========================================================
# CHECK REQUIRED COLUMN
# =========================================================
if "Subscription End Date" not in df.columns:

    st.error(
        "❌ Selected sheet/file mein "
        "`Subscription End Date` column nahi mila."
    )

    st.write("### Available columns:")

    st.write(
        df.columns.tolist()
    )

    st.stop()


# =========================================================
# DATE CLEANING
# =========================================================
df["Subscription End Date"] = pd.to_datetime(
    df["Subscription End Date"],
    errors="coerce"
)


if "Subscription Start Date" in df.columns:

    df["Subscription Start Date"] = pd.to_datetime(
        df["Subscription Start Date"],
        errors="coerce"
    )


# Remove records without valid end date
df = df[
    df["Subscription End Date"].notna()
].copy()


if df.empty:

    st.warning(
        "⚠️ Valid Subscription End Date wala koi record nahi mila."
    )

    st.stop()


# =========================================================
# CREATE RENEWAL MONTH
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


# =========================================================
# AVAILABLE MONTHS
# =========================================================
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


if len(month_options) == 0:

    st.warning(
        "⚠️ Renewal month available nahi hai."
    )

    st.stop()


# =========================================================
# MONTH SELECTION
# =========================================================
st.divider()

st.subheader("📅 Select Renewal Month")


selected_month = st.selectbox(
    "Which month's renewal patients do you want to see?",
    month_options
)


# =========================================================
# FILTER SELECTED MONTH
# =========================================================
filtered_df = df[
    df["Renewal Month"] == selected_month
].copy()


# =========================================================
# NPS PROCESSING
# =========================================================
has_nps = "Latest NPS Score" in filtered_df.columns


if has_nps:

    filtered_df["Latest NPS Score"] = pd.to_numeric(
        filtered_df["Latest NPS Score"],
        errors="coerce"
    )


    def nps_category(score):

        if pd.isna(score):
            return "No NPS"

        elif score >= 9:
            return "🟢 Promoter"

        elif score >= 7:
            return "🟡 Passive"

        else:
            return "🔴 Detractor"


    filtered_df["NPS Category"] = (
        filtered_df["Latest NPS Score"]
        .apply(nps_category)
    )


# =========================================================
# SUMMARY
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
# NPS KPIs
# =========================================================
if has_nps:

    valid_nps = filtered_df[
        filtered_df["Latest NPS Score"].notna()
    ].copy()


    if not valid_nps.empty:

        average_nps = (
            valid_nps[
                "Latest NPS Score"
            ].mean()
        )

        promoters = (
            valid_nps[
                "Latest NPS Score"
            ] >= 9
        ).sum()

        detractors = (
            valid_nps[
                "Latest NPS Score"
            ] <= 6
        ).sum()


        col2.metric(
            "⭐ Average NPS",
            round(average_nps, 1)
        )

        col3.metric(
            "🟢 Promoters",
            int(promoters)
        )

        col4.metric(
            "🔴 Detractors",
            int(detractors)
        )


    else:

        col2.metric(
            "⭐ Average NPS",
            "-"
        )

        col3.metric(
            "🟢 Promoters",
            0
        )

        col4.metric(
            "🔴 Detractors",
            0
        )


else:

    col2.metric(
        "⭐ Average NPS",
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
# PATIENT TABLE
# =========================================================
st.divider()

st.subheader(
    f"📋 {selected_month} Renewal Patients"
)


patient_columns = [
    "Patient ID",
    "Identifier",
    "Patient Name",
    "Email",
    "Subscription Start Date",
    "Subscription End Date",
    "Days Remaining",
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

    display_df = filtered_df[
        available_patient_columns
    ].copy()

else:

    display_df = filtered_df.drop(
        columns=["Renewal Month Sort"],
        errors="ignore"
    ).copy()


display_df = display_df.sort_values(
    "Subscription End Date",
    ascending=True
)


st.dataframe(
    display_df,
    use_container_width=True,
    hide_index=True
)


# =========================================================
# NPS DETAILS
# =========================================================
if has_nps:

    st.divider()

    st.subheader("⭐ NPS Details")


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

    st.info(
        "ℹ️ Uploaded file mein `Latest NPS Score` nahi hai. "
        "Renewal month filtering phir bhi properly work karega."
    )


# =========================================================
# DOWNLOAD SELECTED MONTH
# =========================================================
st.divider()


download_df = filtered_df.drop(
    columns=["Renewal Month Sort"],
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
    label=f"📥 Download {selected_month} Patients",
    data=csv_data,
    file_name=f"{safe_month}_renewal_patients.csv",
    mime="text/csv"
)
