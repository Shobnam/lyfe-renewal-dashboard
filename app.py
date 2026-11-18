uploaded_file = st.file_uploader(
    "📂 Upload Patient File (CSV or Excel)",
    type=["csv", "xlsx"]
)

if uploaded_file is not None:

    try:
        # CSV / Excel load
        if uploaded_file.name.lower().endswith(".csv"):
            df = pd.read_csv(uploaded_file)

        elif uploaded_file.name.lower().endswith(".xlsx"):
            df = pd.read_excel(
                uploaded_file,
                engine="openpyxl"
            )

        st.success(
            f"✅ File uploaded successfully: {uploaded_file.name}"
        )

        # Subscription End Date
        if "Subscription End Date" not in df.columns:
            st.error(
                "File mein `Subscription End Date` column nahi mila."
            )
            st.stop()

        df["Subscription End Date"] = pd.to_datetime(
            df["Subscription End Date"],
            errors="coerce"
        )

        df = df[
            df["Subscription End Date"].notna()
        ].copy()

        # Jan 2026 / Feb 2026 etc.
        df["Renewal Month"] = (
            df["Subscription End Date"]
            .dt.strftime("%b %Y")
        )

        month_table = (
            df[
                ["Subscription End Date", "Renewal Month"]
            ]
            .drop_duplicates(subset=["Renewal Month"])
            .sort_values("Subscription End Date")
        )

        month_options = month_table[
            "Renewal Month"
        ].tolist()

        selected_month = st.selectbox(
            "📅 Select Renewal Month",
            month_options
        )

        filtered_df = df[
            df["Renewal Month"] == selected_month
        ].copy()

        st.success(
            f"✅ {len(filtered_df)} patients found for {selected_month}"
        )

        st.dataframe(
            filtered_df,
            use_container_width=True,
            hide_index=True
        )

    except Exception as e:
        st.error("File process karte waqt error aaya.")
        st.exception(e)

else:
    st.info("👆 CSV ya Excel patient file upload karein.")
