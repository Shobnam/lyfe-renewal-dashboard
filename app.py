import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

st.set_page_config(page_title="Lyfe Renewal Intelligence Engine", page_icon="🩺", layout="wide")
st.title("🩺 Patient Renewal Propensity & Pattern Intelligence")

# ---------------------------------------------------------
# 1. FILE UPLOADER (CSV / EXCEL)
# ---------------------------------------------------------
uploaded_file = st.file_uploader("📂 Apni Patient Export Sheet Upload Karein (CSV ya Excel)", type=["csv", "xlsx"])

if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
        
        # Date Columns Parsing
        for col in ["Plan Start Date", "Plan End Date", "Last Completed Call Date"]:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce")

        # ---------------------------------------------------------
        # 2. SIDEBAR: MONTH FILTER & DYNAMIC WEIGHTAGE SLIDERS
        # ---------------------------------------------------------
        st.sidebar.header("📅 1. Month / Cohort Filter")
        if "Plan End Date" in df.columns:
            df["Renewal_Month"] = df["Plan End Date"].dt.strftime("%Y-%m")
            months = ["All Months"] + sorted(df["Renewal_Month"].dropna().unique().tolist(), reverse=True)
            selected_month = st.sidebar.selectbox("Select Plan End Month:", months)
        else:
            selected_month = "All Months"

        st.sidebar.divider()
        st.sidebar.header("⚖️ 2. Engagement Weightage Sliders")
        st.sidebar.caption("Har factor ka weightage adjust karein:")

        w_lcm = st.sidebar.slider("LCM Call Completion %", 0, 50, 20)
        w_hc = st.sidebar.slider("HC (Health Coach) Completion %", 0, 50, 20)
        w_ct = st.sidebar.slider("CT (Cardio Therapy) Completion %", 0, 50, 15)
        w_psycho = st.sidebar.slider("Psychotherapist Engagement", 0, 30, 10)
        w_clinical = st.sidebar.slider("Clinical Improvement (Wt/SBP/LDL/HbA1c)", 0, 50, 20)
        w_nps = st.sidebar.slider("NPS Score", 0, 30, 15)

        st.sidebar.divider()
        st.sidebar.header("🚨 3. Risk Penalties")
        pen_no_vital = st.sidebar.slider("Penalty: Device Active par No Vitals (Last 30d)", 0, 40, 15)
        pen_inactive_call = st.sidebar.slider("Penalty: Last Call > 60 Days (2 Months)", 0, 50, 25)

        # ---------------------------------------------------------
        # 3. FEATURE ENGINEERING & SCORING ENGINE
        # ---------------------------------------------------------
        calc_df = df.copy()

        def get_num(col_name, default=0.0):
            return pd.to_numeric(calc_df[col_name], errors="coerce").fillna(default) if col_name in calc_df.columns else pd.Series(default, index=calc_df.index)

        lcm_pct = get_num("LCM Completion %")
        hc_pct = get_num("HC Completion %")
        ct_pct = get_num("CT Completion %")
        psycho_pct = get_num("Psychotherapist Completion %")

        wt_change = get_num("Weight Change %")
        sbp_val = get_num("Latest SBP - Plan", 120)
        ldl_val = get_num("Latest LDL - Plan", 100)
        hba1c_val = get_num("Latest HbA1c - Plan", 6.0)

        # Clinical Score Calculation
        clin_score = np.zeros(len(calc_df))
        clin_score += np.where(wt_change > 2.0, 35, np.where(wt_change > 0, 20, 0))
        clin_score += np.where((sbp_val > 0) & (sbp_val <= 130), 25, 0)
        clin_score += np.where((ldl_val > 0) & (ldl_val <= 100), 20, 0)
        clin_score += np.where((hba1c_val > 0) & (hba1c_val <= 6.5), 20, 0)
        clin_score = np.clip(clin_score, 0, 100)

        nps_raw = get_num("Latest NPS Score", 7.0) * 10

        total_weight = max(w_lcm + w_hc + w_ct + w_psycho + w_clinical + w_nps, 1)
        raw_score = (
            (lcm_pct * w_lcm) +
            (hc_pct * w_hc) +
            (ct_pct * w_ct) +
            (psycho_pct * w_psycho) +
            (clin_score * w_clinical) +
            (nps_raw * w_nps)
        ) / total_weight

        # Risk Flag 1: Silent Device
        dev_count = get_num("Active Device Count", 0)
        vitals_30d = get_num("Vital Readings - Last 30 Days of Plan", 0)
        calc_df["Device_No_Vital_Flag"] = np.where((dev_count > 0) & (vitals_30d == 0), "⚠️ Yes (Silent Device)", "OK")

        # Risk Flag 2: Call Gap > 60 Days
        ref_date = calc_df["Plan End Date"].fillna(pd.Timestamp.today())
        if "Last Completed Call Date" in calc_df.columns:
            days_since_call = (ref_date - calc_df["Last Completed Call Date"]).dt.days
        else:
            days_since_call = pd.Series(0, index=calc_df.index)

        calc_df["Days_Since_Last_Call"] = days_since_call.fillna(999)
        calc_df["Call_Gap_2M_Flag"] = np.where(calc_df["Days_Since_Last_Call"] > 60, "🚨 > 60 Days Gap", "Active (<60d)")

        # Penalties application
        penalty_applied = (
            np.where(calc_df["Device_No_Vital_Flag"] != "OK", pen_no_vital, 0) +
            np.where(calc_df["Call_Gap_2M_Flag"] != "Active (<60d)", pen_inactive_call, 0)
        )
        calc_df["Renewal_Score"] = np.clip(raw_score - penalty_applied, 0, 100).round(1)

        calc_df["Renewal_Chance"] = pd.cut(
            calc_df["Renewal_Score"],
            bins=[-1, 40, 70, 101],
            labels=["🔴 Low Chance", "🟡 Moderate Chance", "🟢 High Chance"]
        )

        # Apply Month Filter
        if selected_month != "All Months":
            view_df = calc_df[calc_df["Renewal_Month"] == selected_month].copy()
        else:
            view_df = calc_df.copy()

        # ---------------------------------------------------------
        # 4. TABS STRUCTURE
        # ---------------------------------------------------------
        tab1, tab2, tab3 = st.tabs([
            "📊 Cohort Prediction & Risk", 
            "🧠 Historical Patterns", 
            "🎯 Patient Pitch & Success Stories"
        ])

        with tab1:
            st.subheader(f"Cohort Overview: {selected_month}")
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Total Patients", len(view_df))
            c2.metric("🟢 High Chance", len(view_df[view_df["Renewal_Chance"] == "🟢 High Chance"]))
            c3.metric("🔴 Low Chance", len(view_df[view_df["Renewal_Chance"] == "🔴 Low Chance"]))
            c4.metric("⚠️ Silent Devices", len(view_df[view_df["Device_No_Vital_Flag"] != "OK"]))
            c5.metric("🚨 Call Gap > 2M", len(view_df[view_df["Call_Gap_2M_Flag"] != "Active (<60d)"]))

            st.divider()

            show_cols = [
                "Identifier", "Patient Name", "Plan End Date", "Renewal_Score", "Renewal_Chance",
                "Call_Gap_2M_Flag", "Device_No_Vital_Flag", "LCM Completion %", "HC Completion %",
                "Weight Change %", "Latest NPS Score", "Renewed Within 30 Days"
            ]
            st.dataframe(view_df[[c for c in show_cols if c in view_df.columns]].sort_values(by="Renewal_Score", ascending=False), use_container_width=True)

        with tab2:
            st.subheader("🔍 Historical Pattern: Renewed vs Churned Behaviour")
            if "Renewed Within 30 Days" in df.columns:
                pattern_df = df[df["Renewed Within 30 Days"].isin([0, 1])].copy()
                if not pattern_df.empty:
                    pattern_df["Status"] = pattern_df["Renewed Within 30 Days"].map({1: "✅ Renewed", 0: "❌ Churned"})
                    metrics = ["LCM Completion %", "HC Completion %", "CT Completion %", "Weight Change %", "Latest NPS Score"]
                    avail_m = [m for m in metrics if m in pattern_df.columns]
                    summary = pattern_df.groupby("Status")[avail_m].mean().round(2).T
                    st.dataframe(summary, use_container_width=True)
                else:
                    st.warning("Mature renewal records available nahi hain.")
            else:
                st.warning("`Renewed Within 30 Days` column missing hai.")

        with tab3:
            st.subheader("🎯 Patient Pitch & Converted Success Stories")
            sub_tab1, sub_tab2 = st.tabs(["📞 Patients to Pitch (At-Risk / Expiring)", "✅ Converted Success Stories"])
            
            with sub_tab1:
                st.write("**Yeh woh patients hain jinko call/pitch karni hai:**")
                pitch_df = view_df[view_df["Renewed Within 30 Days"] != 1].copy()
                
                def make_script(row):
                    name = row.get("Patient Name", "Patient")
                    lcm_c = row.get("LCM Calls Completed", 0)
                    wt = row.get("Weight Change %", 0)
                    return f"Hi {name}, aapne is plan mein {lcm_c} LCM calls complete ki hain aur weight {wt}% manage kiya hai. Next plan mein hum ise aur better karenge!"

                if not pitch_df.empty:
                    pitch_df["Suggested Pitch Hook"] = pitch_df.apply(make_script, axis=1)
                    cols = ["Identifier", "Patient Name", "Plan End Date", "Renewal_Score", "Renewal_Chance", "Suggested Pitch Hook", "Latest NPS Comments"]
                    st.dataframe(pitch_df[[c for c in cols if c in pitch_df.columns]], use_container_width=True)
                    st.download_button("📥 Download Pitch List", pitch_df.to_csv(index=False).encode("utf-8"), "pitch_list.csv", "text/csv")
                else:
                    st.success("Koi at-risk patient nahi hai!")

            with sub_tab2:
                st.write("**Successful Renewals (Inhe reference ke taur par use karein):**")
                success_df = view_df[view_df["Renewed Within 30 Days"] == 1].copy()
                if not success_df.empty:
                    s_cols = ["Identifier", "Patient Name", "Plan Name", "Plan End Date", "LCM Completion %", "HC Completion %", "Weight Change %", "Latest NPS Score", "Latest NPS Comments"]
                    st.dataframe(success_df[[c for c in s_cols if c in success_df.columns]], use_container_width=True)
                else:
                    st.warning("Is selection mein koi renewal record nahi hai.")

    except Exception as e:
        st.error(f"Error processing file: {e}")
else:
    st.info("👆 Shuru karne ke liye upar apni `patients_data.csv` file upload karein.")