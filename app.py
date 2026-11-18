import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="Lyfe Renewal App",
    page_icon="🩺",
    layout="wide"
)

st.title("🩺 Lyfe Renewal App")

# ---------------------------------------------------------
# STEP 1: SELECT RENEWAL MONTH
# ---------------------------------------------------------

st.subheader("Select Renewal Month")

renewal_month = st.selectbox(
    "Which month's renewal patients do you want to see?",
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

st.success(f"Selected Renewal Month: {renewal_month}")
