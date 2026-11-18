st.sidebar.header("📅 Renewal Cohort")

if "Plan End Date" in df.columns:
    df["Plan End Date"] = pd.to_datetime(
        df["Plan End Date"],
        errors="coerce"
    )

    df["Renewal_Month"] = df["Plan End Date"].dt.strftime("%b %Y")

    available_months = (
        df["Plan End Date"]
        .dropna()
        .dt.to_period("M")
        .sort_values()
        .unique()
    )

    month_options = [
        month.strftime("%b %Y")
        for month in available_months
    ]

    selected_month = st.sidebar.selectbox(
        "Select Renewal Month",
        month_options
    )

    view_df = df[
        df["Renewal_Month"] == selected_month
    ].copy()

    st.sidebar.success(
        f"{len(view_df)} patients in {selected_month}"
    )

else:
    st.error("`Plan End Date` column file mein nahi mila.")
    st.stop()
