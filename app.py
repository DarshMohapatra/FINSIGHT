
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import joblib
import os
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from groq import Groq

# ── Page Config ───────────────────────────────────────
st.set_page_config(
    page_title="FinSight — AI Finance",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── Custom CSS ────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Mono:wght@400;500&display=swap');

* { font-family: 'Syne', sans-serif !important; }
.stApp { background: #020408; color: #e8eaf0; }
.stButton > button {
    background: linear-gradient(135deg, #00f5a0, #00d4ff) !important;
    color: #000 !important;
    font-weight: 700 !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 12px 28px !important;
    font-size: 15px !important;
    width: 100% !important;
}
.stButton > button:hover { opacity: 0.9 !important; }
.stTextInput > div > div > input {
    background: #0a0e1a !important;
    color: white !important;
    border: 1px solid #1a1f2e !important;
    border-radius: 10px !important;
}
.stFileUploader {
    background: #0a0e1a !important;
    border: 1px dashed #1a1f2e !important;
    border-radius: 16px !important;
}
div[data-testid="stMetricValue"] {
    color: #00f5a0 !important;
    font-size: 28px !important;
    font-weight: 800 !important;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    color: #666 !important;
    font-weight: 600 !important;
}
.stTabs [aria-selected="true"] {
    color: #00f5a0 !important;
    border-bottom: 2px solid #00f5a0 !important;
}
.metric-card {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 16px;
    padding: 24px;
    text-align: center;
}
.metric-value {
    font-size: 32px;
    font-weight: 800;
    background: linear-gradient(135deg, #00f5a0, #00d4ff);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-family: "DM Mono", monospace !important;
}
.metric-label {
    font-size: 12px;
    color: #666;
    font-family: "DM Mono", monospace !important;
    letter-spacing: 1px;
    text-transform: uppercase;
    margin-top: 8px;
}
.chat-msg-user {
    background: linear-gradient(135deg, #00f5a0, #00d4ff);
    color: #000;
    padding: 14px 18px;
    border-radius: 16px 16px 4px 16px;
    margin: 8px 0;
    font-weight: 500;
    max-width: 80%;
    margin-left: auto;
}
.chat-msg-ai {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.08);
    color: #e8eaf0;
    padding: 14px 18px;
    border-radius: 16px 16px 16px 4px;
    margin: 8px 0;
    max-width: 80%;
}
.hero-title {
    font-size: 64px;
    font-weight: 800;
    letter-spacing: -3px;
    line-height: 1.0;
}
.hero-gradient {
    background: linear-gradient(135deg, #00f5a0, #00d4ff, #7b61ff);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.section-divider {
    border: none;
    border-top: 1px solid #1a1f2e;
    margin: 32px 0;
}
footer { display: none !important; }
#MainMenu { display: none !important; }
</style>
""", unsafe_allow_html=True)

# ── Session State ─────────────────────────────────────
if "user_df" not in st.session_state:
    st.session_state.user_df = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "analysis_done" not in st.session_state:
    st.session_state.analysis_done = False

# ── Groq Client ───────────────────────────────────────
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
if GROQ_API_KEY:
    groq_client = Groq(api_key=GROQ_API_KEY)

# ── Categorizer ───────────────────────────────────────
def categorize_transaction(description):
    desc = str(description).upper()
    if any(k in desc for k in ["SALARY","SAL","PAYROLL","INDO GIBL","GIBL"]):
        return "Salary"
    elif any(k in desc for k in ["TRF FROM","TRF FRM","TRANSFER IN","INTERNAL FUND"]):
        return "Transfer In"
    elif any(k in desc for k in ["TRF TO","TRANSFER OUT","FDRL/INTERNAL FUND TRANSFE"]):
        return "Transfer Out"
    elif any(k in desc for k in ["NEFT","RTGS","FDRL/NATIONAL ELECTRONIC","IMPS"]):
        return "Online Payment"
    elif any(k in desc for k in ["CASHDEP","CASH DEP","CASH DEPOSIT"]):
        return "Cash Deposit"
    elif any(k in desc for k in ["CASHWDL","CASH WDL","CASH WITHDRAWAL","ATM"]):
        return "ATM/Cash Withdrawal"
    elif "UPI" in desc:
        return "UPI Payment"
    elif any(k in desc for k in ["CHQ","CHEQUE","CHECK"]):
        return "Cheque"
    elif any(k in desc for k in ["EMI","LOAN","MORTGAGE"]):
        return "Loan/EMI"
    elif any(k in desc for k in ["TAX","GST","GOVT","GOVERNMENT"]):
        return "Tax/Government"
    elif "INDIAFORENSIC" in desc:
        return "Business Transfer"
    else:
        return "Other"

# ── Process Upload ────────────────────────────────────
def process_file(uploaded_file):
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    df.columns = df.columns.str.strip()

    # Drop junk columns
    junk = [c for c in df.columns if c in [".", "CHQ.NO.", "VALUE DATE"]]
    df.drop(columns=junk, inplace=True)

    # Clean
    df["DATE"] = pd.to_datetime(df["DATE"], errors="coerce")
    df["WITHDRAWAL AMT"] = df["WITHDRAWAL AMT"].fillna(0)
    df["DEPOSIT AMT"] = df["DEPOSIT AMT"].fillna(0)
    df["BALANCE AMT"] = df["BALANCE AMT"].fillna(0)
    df = df[(df["WITHDRAWAL AMT"] > 0) | (df["DEPOSIT AMT"] > 0)]
    df.dropna(subset=["TRANSACTION DETAILS"], inplace=True)

    # Categorize
    df["CATEGORY"] = df["TRANSACTION DETAILS"].apply(categorize_transaction)

    # Anomaly detection
    features = df[["WITHDRAWAL AMT","DEPOSIT AMT","BALANCE AMT"]].copy()
    scaler = StandardScaler()
    fs = scaler.fit_transform(features)
    model = IsolationForest(n_estimators=100, contamination=0.02, random_state=42)
    model.fit(fs)
    df["ANOMALY_SCORE"] = model.decision_function(fs)
    df["IS_ANOMALY"] = model.predict(fs)
    df["IS_ANOMALY"] = df["IS_ANOMALY"].map({1: 0, -1: 1})
    df["MONTH"] = df["DATE"].dt.to_period("M")

    return df

# ── Header ────────────────────────────────────────────
st.markdown("""
<div style="padding: 48px 0 24px 0;">
    <div class="hero-title">
        <span>Fin</span><span class="hero-gradient">Sight</span>
    </div>
    <p style="font-size:18px; color:#666; margin-top:12px; font-family:'DM Mono',monospace;">
        AI-POWERED FINANCIAL INTELLIGENCE
    </p>
</div>
<hr class="section-divider">
""", unsafe_allow_html=True)

# ── Tabs ──────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "📤  Upload", 
    "📊  Dashboard", 
    "📈  Forecast", 
    "🤖  AI Advisor"
])

# ════════════════════════════════════════════════════════
# TAB 1 — UPLOAD
# ════════════════════════════════════════════════════════
with tab1:
    st.markdown("### Upload Your Bank Statement")
    st.markdown("<p style='color:#666; font-family:DM Mono,monospace; font-size:13px;'>SUPPORTS CSV AND EXCEL (.XLSX) FORMATS</p>", unsafe_allow_html=True)

    uploaded_file = st.file_uploader("", type=["csv", "xlsx"], label_visibility="collapsed")

    if uploaded_file and st.button("⚡ Analyze My Finances"):
        with st.spinner("🔍 Analyzing your transactions..."):
            try:
                df = process_file(uploaded_file)
                st.session_state.user_df = df
                st.session_state.analysis_done = True
                st.success("✅ Analysis Complete!")
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")

    if st.session_state.analysis_done and st.session_state.user_df is not None:
        df = st.session_state.user_df

        st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

        # Metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{len(df):,}</div>
                <div class="metric-label">Total Transactions</div>
            </div>""", unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">₹{df["WITHDRAWAL AMT"].sum()/1e7:.0f}Cr</div>
                <div class="metric-label">Total Withdrawn</div>
            </div>""", unsafe_allow_html=True)
        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{int(df["IS_ANOMALY"].sum()):,}</div>
                <div class="metric-label">Anomalies Flagged</div>
            </div>""", unsafe_allow_html=True)
        with col4:
            coverage = f"{(df['CATEGORY'] != 'Other').sum() / len(df) * 100:.1f}%"
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{coverage}</div>
                <div class="metric-label">Auto Categorized</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("### 📋 Sample Transactions")
        st.dataframe(
            df[["DATE","TRANSACTION DETAILS","WITHDRAWAL AMT","DEPOSIT AMT","CATEGORY","IS_ANOMALY"]].head(20),
            use_container_width=True
        )

# ════════════════════════════════════════════════════════
# TAB 2 — DASHBOARD
# ════════════════════════════════════════════════════════
with tab2:
    if st.session_state.user_df is None:
        st.warning("⚠️ Please upload your bank statement first in the Upload tab!")
    else:
        df = st.session_state.user_df
        st.markdown("### Your Financial Dashboard")

        def crore_fmt(x, pos): return f"₹{x/1e7:.0f}Cr"

        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        fig.patch.set_facecolor("#020408")
        for ax in axes.flat:
            ax.set_facecolor("#0a0e1a")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["bottom"].set_color("#1a1f2e")
            ax.spines["left"].set_color("#1a1f2e")
            ax.tick_params(colors="#666")

        fig.suptitle("FinSight — Financial Dashboard",
                     fontsize=18, fontweight="bold", color="white", y=0.98)

        # 1. Monthly Withdrawals
        monthly_wd = df.groupby(df["DATE"].dt.to_period("M"))["WITHDRAWAL AMT"].sum()
        axes[0,0].plot(range(len(monthly_wd)), monthly_wd.values, color="#00f5a0", linewidth=2)
        axes[0,0].fill_between(range(len(monthly_wd)), monthly_wd.values, alpha=0.1, color="#00f5a0")
        axes[0,0].set_title("Monthly Withdrawals", color="white", fontsize=13)
        axes[0,0].yaxis.set_major_formatter(mticker.FuncFormatter(crore_fmt))
        step = max(1, len(monthly_wd)//8)
        axes[0,0].set_xticks(range(0, len(monthly_wd), step))
        axes[0,0].set_xticklabels(monthly_wd.index.astype(str)[::step], rotation=45, ha="right", color="#666", fontsize=8)

        # 2. Category Bar
        cat_counts = df["CATEGORY"].value_counts()
        colors = ["#00f5a0","#00d4ff","#7b61ff","#ff4d6d","#ffd60a",
                  "#ff9f43","#a8e063","#f8a5c2","#778ca3","#2d98da","#4b7bec","#a55eea"]
        axes[0,1].barh(cat_counts.index, cat_counts.values, color=colors[:len(cat_counts)])
        axes[0,1].set_title("Transactions by Category", color="white", fontsize=13)
        axes[0,1].tick_params(colors="#aaa", labelsize=9)
        axes[0,1].invert_yaxis()

        # 3. Anomaly Timeline
        anomaly_monthly = df[df["IS_ANOMALY"]==1].groupby("MONTH")["IS_ANOMALY"].count()
        x = list(range(len(anomaly_monthly)))
        axes[1,0].bar(x, anomaly_monthly.values, color="#ff4d6d", alpha=0.8)
        axes[1,0].set_title("Anomalies Per Month", color="white", fontsize=13)
        step2 = max(1, len(x)//8)
        axes[1,0].set_xticks(x[::step2])
        axes[1,0].set_xticklabels(anomaly_monthly.index.astype(str)[::step2], rotation=45, ha="right", color="#666", fontsize=8)

        # 4. Distribution
        wd = df[df["WITHDRAWAL AMT"] > 0]["WITHDRAWAL AMT"]
        axes[1,1].hist(wd, bins=50, color="#00d4ff", edgecolor="#020408", log=True)
        axes[1,1].set_title("Withdrawal Distribution (log)", color="white", fontsize=13)
        axes[1,1].tick_params(colors="#666")

        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

# ════════════════════════════════════════════════════════
# TAB 3 — FORECAST
# ════════════════════════════════════════════════════════
with tab3:
    if st.session_state.user_df is None:
        st.warning("⚠️ Please upload your bank statement first in the Upload tab!")
    else:
        df = st.session_state.user_df
        st.markdown("### 6-Month Expense Forecast")
        st.markdown("<p style='color:#666;'>Powered by Facebook Prophet time series forecasting</p>", unsafe_allow_html=True)

        with st.spinner("📈 Generating forecast..."):
            from prophet import Prophet

            monthly_spend = df.groupby(
                df["DATE"].dt.to_period("M"))["WITHDRAWAL AMT"].sum().reset_index()
            monthly_spend.columns = ["ds", "y"]
            monthly_spend["ds"] = monthly_spend["ds"].dt.to_timestamp()

            m = Prophet(yearly_seasonality=True, weekly_seasonality=False,
                        daily_seasonality=False, changepoint_prior_scale=0.05)
            m.fit(monthly_spend)
            future = m.make_future_dataframe(periods=6, freq="MS")
            forecast = m.predict(future)

        def crore_fmt(x, pos): return f"₹{x/1e7:.0f}Cr"

        fig, axes = plt.subplots(2, 1, figsize=(14, 10))
        fig.patch.set_facecolor("#020408")
        for ax in axes:
            ax.set_facecolor("#0a0e1a")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["bottom"].set_color("#1a1f2e")
            ax.spines["left"].set_color("#1a1f2e")

        fig.suptitle("FinSight — Expense Forecast", fontsize=16, fontweight="bold", color="white")

        axes[0].plot(monthly_spend["ds"], monthly_spend["y"], color="#00d4ff", linewidth=2, label="Actual")
        axes[0].plot(forecast["ds"], forecast["yhat"], color="#00f5a0", linewidth=2, linestyle="--", label="Forecast")
        axes[0].fill_between(forecast["ds"], forecast["yhat_lower"], forecast["yhat_upper"], alpha=0.15, color="#00f5a0")
        axes[0].axvline(x=monthly_spend["ds"].max(), color="#ff4d6d", linestyle="--", linewidth=1.5, label="Forecast Start")
        axes[0].set_title("6-Month Withdrawal Forecast", color="white", fontsize=13)
        axes[0].yaxis.set_major_formatter(mticker.FuncFormatter(crore_fmt))
        axes[0].legend(facecolor="#0a0e1a", labelcolor="white")
        axes[0].tick_params(colors="#666")

        future_only = forecast.tail(6)
        axes[1].bar(future_only["ds"].dt.strftime("%Y-%m"), future_only["yhat"], color="#7b61ff", alpha=0.85)
        axes[1].set_title("Predicted Spending — Next 6 Months", color="white", fontsize=13)
        axes[1].yaxis.set_major_formatter(mticker.FuncFormatter(crore_fmt))
        axes[1].tick_params(colors="#aaa")

        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        # Show table
        st.markdown("### Predicted Values")
        future_table = future_only[["ds","yhat","yhat_lower","yhat_upper"]].copy()
        future_table["Month"] = future_table["ds"].dt.strftime("%Y-%m")
        future_table["Predicted (₹ Cr)"] = (future_table["yhat"]/1e7).round(2)
        future_table["Lower (₹ Cr)"] = (future_table["yhat_lower"]/1e7).round(2)
        future_table["Upper (₹ Cr)"] = (future_table["yhat_upper"]/1e7).round(2)
        st.dataframe(future_table[["Month","Predicted (₹ Cr)","Lower (₹ Cr)","Upper (₹ Cr)"]], use_container_width=True)

# ════════════════════════════════════════════════════════
# TAB 4 — AI ADVISOR
# ════════════════════════════════════════════════════════
with tab4:
    st.markdown("### 🤖 AI Financial Advisor")
    st.markdown("<p style='color:#666; font-family:DM Mono,monospace; font-size:13px;'>POWERED BY LLAMA 3.3 70B — ANSWERS BASED ON YOUR UPLOADED DATA</p>", unsafe_allow_html=True)

    if st.session_state.user_df is None:
        st.warning("⚠️ Please upload your bank statement first in the Upload tab!")
    elif not GROQ_API_KEY:
        st.error("❌ Groq API key not set. Add it as GROQ_API_KEY in Streamlit secrets.")
    else:
        df = st.session_state.user_df

        # Build context from uploaded data
        total_wd = df["WITHDRAWAL AMT"].sum() / 1e7
        total_dep = df["DEPOSIT AMT"].sum() / 1e7
        anomaly_count = int(df["IS_ANOMALY"].sum())
        avg_monthly = df.groupby("MONTH")["WITHDRAWAL AMT"].sum().mean() / 1e7
        top_cat = df["CATEGORY"].value_counts().index[0]
        cat_text = "
".join([f"- {cat}: ₹{amt/1e7:.1f}Cr"
                               for cat, amt in df.groupby("CATEGORY")["WITHDRAWAL AMT"].sum().sort_values(ascending=False).items()])

        context = f"""You are FinSight, a personal AI financial advisor.
Analyze THIS user's actual uploaded bank data:
- Total Withdrawals: ₹{total_wd:.1f} Crores
- Total Deposits: ₹{total_dep:.1f} Crores
- Most Common Transaction: {top_cat}
- Suspicious Transactions: {anomaly_count}
- Avg Monthly Spending: ₹{avg_monthly:.1f} Crores
- Total Transactions: {len(df):,}
Spending by Category:
{cat_text}
Give personalized, data-backed advice. Be concise and friendly."""

        # Display chat history
        for msg in st.session_state.chat_history:
            if msg["role"] == "user":
                st.markdown(f'<div class="chat-msg-user">{msg["content"]}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="chat-msg-ai">{msg["content"]}</div>', unsafe_allow_html=True)

        # Example questions
        st.markdown("<br>", unsafe_allow_html=True)
        cols = st.columns(3)
        examples = [
            "What is my biggest spending category?",
            "How many suspicious transactions do I have?",
            "Give me 3 tips to reduce my spending!"
        ]
        for i, ex in enumerate(examples):
            with cols[i]:
                if st.button(ex, key=f"ex_{i}"):
                    st.session_state.chat_history.append({"role": "user", "content": ex})
                    response = groq_client.chat.completions.create(
                        model="llama-3.3-70b-versatile",
                        messages=[{"role": "system", "content": context},
                                  {"role": "user", "content": ex}]
                    )
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": response.choices[0].message.content
                    })
                    st.rerun()

        # Chat input
        user_input = st.chat_input("Ask anything about your finances...")
        if user_input:
            st.session_state.chat_history.append({"role": "user", "content": user_input})
            response = groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "system", "content": context},
                          {"role": "user", "content": user_input}]
            )
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": response.choices[0].message.content
            })
            st.rerun()
