import streamlit as st
import pandas as pd

# ---------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------
st.set_page_config(
    page_title="Lyfe Renewal App",
    page_icon="🩺",
    layout="wide"
)

st.title("🩺 Lyfe Renewal App")


# ---------------------------------------------------------
# STEP 1: CSV UPLOAD
# ---------------------------------------------------------
uploaded_file = st.file_uploader(
    "📂 Upload Patient CSV",
    type=["csv"]
)


if uploaded_file is not None:

    try:
        # Load CSV
        df = pd.read_csv(uploaded_file)

        # ---------------------------------------------------------
        # CHECK SUBSCRIPTION END DATE COLUMN
        # ---------------------------------------------------------
        if "Subscription End Date" not in df.columns:
            st.error(
                "CSV mein `Subscription End Date` column nahi mila."
            )

            st.write("Available columns:")
            st.write(df.columns.tolist())

            st.stop()


        # Convert to date
        df["Subscription End Date"] = pd.to_datetime(
            df["Subscription End Date"],
            errors="coerce"
        )

        # Remove rows where date is invalid/missing
        df = df[
            df["Subscription End Date"].notna()
        ].copy()


        # ---------------------------------------------------------
        # CREATE MONTH-YEAR
        # Example: Jan 2026
        # ---------------------------------------------------------
        df["Renewal Month"] = (
            df["Subscription End Date"]
            .dt.strftime("%b %Y")
        )


        # ---------------------------------------------------------
        # MONTHS AVAILABLE IN CSV
        # ---------------------------------------------------------
        month_table = (
            df[
                ["Subscription End Date", "Renewal Month"]
            ]
            .drop_duplicates(subset=["Renewal Month"])
            .sort_values("Subscription End Date")
        )

        month_options = (
            month_table["Renewal Month"]
            .tolist()
        )


        # ---------------------------------------------------------
        # STEP 2: SELECT RENEWAL MONTH
        # ---------------------------------------------------------
        st.subheader("📅 Select Renewal Month")

        selected_month = st.selectbox(
            "Select Month",
            month_options
        )


        # ---------------------------------------------------------
        # FILTER SELECTED MONTH
        # ---------------------------------------------------------
        filtered_df = df[
            df["Renewal Month"] == selected_month
        ].copy()


        # ---------------------------------------------------------
        # HEADER
        # ---------------------------------------------------------
        st.success(
            f"{len(filtered_df)} patients found whose plan ends in "
            f"{selected_month}"
        )


        # ---------------------------------------------------------
        # SUMMARY METRICS
        # ---------------------------------------------------------
        col1, col2, col3, col4 = st.columns(4)

        col1.metric(
            "Total Renewal Patients",
            len(filtered_df)
        )


        # NPS Average
        if "Latest NPS Score" in filtered_df.columns:

            filtered_df["Latest NPS Score"] = pd.to_numeric(
                filtered_df["Latest NPS Score"],
                errors="coerce"
            )

            avg_nps = filtered_df[
                "Latest NPS Score"
            ].mean()

            col2.metric(
                "Average NPS",
                round(avg_nps, 1)
                if pd.notna(avg_nps)
                else "-"
            )

        else:
            col2.metric(
                "Average NPS",
                "-"
            )


        # Promoters
        if "Latest NPS Score" in filtered_df.columns:

            promoters = (
                filtered_df["Latest NPS Score"] >= 9
            ).sum()

            col3.metric(
                "NPS Promoters",
                int(promoters)
            )

        else:
            col3.metric(
                "NPS Promoters",
                "-"
            )


        # Detractors
        if "Latest NPS Score" in filtered_df.columns:

            detractors = (
                filtered_df["Latest NPS Score"] <= 6
            ).sum()

            col4.metric(
                "NPS Detractors",
                int(detractors)
            )

        else:
            col4.metric(
                "NPS Detractors",
                "-"
            )


        # ---------------------------------------------------------
        # NPS CATEGORY
        # ---------------------------------------------------------
        if "Latest NPS Score" in filtered_df.columns:

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


        # ---------------------------------------------------------
        # PATIENT TABLE
        # ---------------------------------------------------------
        st.divider()

        st.subheader(
            f"📋 {selected_month} Renewal Patients"
        )


        display_columns = [
            "Identifier",
            "Patient Name",
            "Subscription Start Date",
            "Subscription End Date",
            "Latest Engagement Status",
            "HC Name",
            "LCM Name",
            "Latest NPS Score",
            "NPS Category",
            "Latest NPS Comments"
        ]


        available_columns = [
            col
            for col in display_columns
            if col in filtered_df.columns
        ]


        st.dataframe(
            filtered_df[available_columns],
            use_container_width=True,
            hide_index=True
        )


        # ---------------------------------------------------------
        # NPS DETAILS
        # ---------------------------------------------------------
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
                "Overall Service Feedback",
                "Health Coach Feedback",
                "LCM Feedback",
                "Cardio Therapy Feedback",
                "Monitoring Devices Feedback",
                "Lyfe App Feedback",
                "Alert Management Feedback"
            ]


            available_nps_columns = [
                col
                for col in nps_columns
                if col in filtered_df.columns
            ]


            nps_df = filtered_df[
                available_nps_columns
            ].copy()


            st.dataframe(
                nps_df,
                use_container_width=True,
                hide_index=True
            )

        else:

            st.warning(
                "`Latest NPS Score` column CSV mein nahi mila."
            )


    except Exception as e:

        st.error(
            "CSV process karte waqt error aaya."
        )

        st.exception(e)


else:

    st.info(
        "👆 Patient CSV upload karein."
    )
