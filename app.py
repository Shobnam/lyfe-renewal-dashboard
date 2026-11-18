import streamlit as st
import pandas as pd
from sqlalchemy import create_engine, text
from urllib.parse import quote_plus
from pathlib import Path


# ---------------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------------
st.set_page_config(
    page_title="Lyfe Renewal App",
    page_icon="🩺",
    layout="wide"
)

st.title("🩺 Lyfe Renewal App")


# ---------------------------------------------------------
# DATABASE CONNECTION
# ---------------------------------------------------------
@st.cache_resource
def get_database_engine():

    db_host = st.secrets["postgres"]["host"]
    db_port = st.secrets["postgres"]["port"]
    db_name = st.secrets["postgres"]["database"]
    db_user = st.secrets["postgres"]["user"]
    db_password = quote_plus(
        st.secrets["postgres"]["password"]
    )

    connection_url = (
        f"postgresql+psycopg2://"
        f"{db_user}:{db_password}@"
        f"{db_host}:{db_port}/"
        f"{db_name}?sslmode=require"
    )

    return create_engine(
        connection_url,
        pool_pre_ping=True
    )


# ---------------------------------------------------------
# STEP 1: RENEWAL MONTH SELECTION
# ---------------------------------------------------------
renewal_month = st.selectbox(
    "Select Renewal Month",
    [
        "Jan 2026",
        "Feb 2026",
        "Mar 2026",
        "Apr 2026",
        "May 2026",
        "Jun 2026",
        "Jul 2026",
        "Aug 2026",
        "Sep 2026",
        "Oct 2026",
        "Nov 2026",
        "Dec 2026"
    ]
)


# Convert selected month into start and end dates
month_start = pd.to_datetime(
    renewal_month,
    format="%b %Y"
).date()

next_month_start = (
    pd.Timestamp(month_start)
    + pd.DateOffset(months=1)
).date()


# ---------------------------------------------------------
# LOAD SQL FROM query.sql
# ---------------------------------------------------------
query_file = Path(__file__).parent / "query.sql"

if not query_file.exists():
    st.error("query.sql file GitHub repository mein nahi mili.")
    st.stop()

sql_query = query_file.read_text(encoding="utf-8")


# ---------------------------------------------------------
# LOAD SELECTED MONTH PATIENTS
# ---------------------------------------------------------
try:

    engine = get_database_engine()

    with st.spinner(
        f"{renewal_month} renewal patients load ho rahe hain..."
    ):

        with engine.connect() as connection:

            df = pd.read_sql_query(
                text(sql_query),
                connection,
                params={
                    "month_start": month_start,
                    "next_month_start": next_month_start
                }
            )

    st.success(
        f"{len(df)} patients found whose plan ends in {renewal_month}"
    )

    if df.empty:

        st.warning(
            f"{renewal_month} mein kisi patient ka plan end nahi ho raha."
        )

    else:

        # Summary
        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Total Renewal Patients",
            len(df)
        )

        if "Subscription End Date" in df.columns:

            end_dates = pd.to_datetime(
                df["Subscription End Date"],
                errors="coerce"
            )

            col2.metric(
                "First Plan End Date",
                end_dates.min().strftime("%d-%b-%Y")
                if end_dates.notna().any()
                else "-"
            )

            col3.metric(
                "Last Plan End Date",
                end_dates.max().strftime("%d-%b-%Y")
                if end_dates.notna().any()
                else "-"
            )

        st.subheader(
            f"📋 {renewal_month} Renewal Patients"
        )

        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True
        )

except Exception as error:

    st.error("Database se data load nahi hua.")

    # Development ke time exact error dikhayega
    st.exception(error)
