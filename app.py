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

st.set_page_config(
    page_title="FinSight — AI Finance",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;700;900&family=DM+Mono:wght@400;500&display=swap');
* { font-family: 'Syne', sans-serif !important; }
.stApp { background: #050810 !important; color: #e0e4f0 !important; }
.stButton > button {
    background: linear-gradient(135deg, #00f5a0, #00d4ff) !important;
    color: #000 !important; font-weight: 700 !important;
    border: none !important; border-radius: 10px !important;
    padding: 12px 28px !important; width: 100% !important;
}
div[data-testid="stMetricValue"] {
    color: #00f5a0 !important; font-size: 28px !important; font-weight: 800 !important;
}
.stTabs [data-baseweb="tab"] { background: transparent !important; color: #555 !important; font-weight: 700 !important; }
.stTabs [aria-selected="true"] { color: #00f5a0 !important; border-bottom: 2px solid #00f5a0 !important; }
.stTabs [data-baseweb="tab-list"] { background: transparent !important; border-bottom: 1px solid #1a1f2e !important; }
.metric-card { background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.07); border-radius: 16px; padding: 24px; text-align: center; }
.metric-value { font-size: 30px; font-weight: 900; background: linear-gradient(135deg, #00f5a0, #00d4ff); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-family: 'DM Mono', monospace !important; }
.metric-label { font-size: 10px; color: #555; font-family: 'DM Mono', monospace !important; letter-spacing: 2px; text-transform: uppercase; margin-top: 6px; }
.chat-user { background: linear-gradient(135deg, #00f5a0, #00d4ff); color: #000; padding: 14px 18px; border-radius: 16px 16px 4px 16px; margin: 8px 0 8px auto; font-weight: 600; max-width: 80%; font-size: 14px; }
.chat-ai { background: rgba(255,255,255,0.04); border: 1px solid rgba(255,255,255,0.08); color: #e0e4f0; padding: 14px 18px; border-radius: 16px 16px 16px 4px; margin: 8px auto 8px 0; max-width: 80%; font-size: 14px; line-height: 1.6; }
footer { display: none !important; }
#MainMenu { display: none !important; }
header { display: none !important; }
</style>
""", unsafe_allow_html=True)

if "user_df" not in st.session_state:
    st.session_state.user_df = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "analysis_done" not in st.session_state:
    st.session_state.analysis_done = False

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
groq_client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None

def categorize_transaction(description):
    desc = str(description).upper()
    if any(k in desc for k in ["SALARY","SAL","PAYROLL","INDO GIBL","GIBL"]): return "Salary"
    elif any(k in desc for k in ["TRF FROM","TRF FRM","TRANSFER IN","INTERNAL FUND"]): return "Transfer In"
    elif any(k in desc for k in ["TRF TO","TRANSFER OUT","FDRL/INTERNAL FUND TRANSFE"]): return "Transfer Out"
    elif any(k in desc for k in ["NEFT","RTGS","FDRL/NATIONAL ELECTRONIC","IMPS"]): return "Online Payment"
    elif any(k in desc for k in ["CASHDEP","CASH DEP","CASH DEPOSIT"]): return "Cash Deposit"
    elif any(k in desc for k in ["CASHWDL","CASH WDL","CASH WITHDRAWAL","ATM"]): return "ATM/Cash Withdrawal"
    elif "UPI" in desc: return "UPI Payment"
    elif any(k in desc for k in ["CHQ","CHEQUE","CHECK"]): return "Cheque"
    elif any(k in desc for k in ["EMI","LOAN","MORTGAGE"]): return "Loan/EMI"
    elif any(k in desc for k in ["TAX","GST","GOVT","GOVERNMENT"]): return "Tax/Government"
    elif "INDIAFORENSIC" in desc: return "Business Transfer"
    else: return "Other"

def process_file(uploaded_file):
    df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith(".csv") else pd.read_excel(uploaded_file)
    df.columns = df.columns.str.strip()
    for col in ["WITHDRAWAL AMT","DEPOSIT AMT","BALANCE AMT"]:
        if col not in df.columns: df[col] = 0
    df["DATE"] = pd.to_datetime(df["DATE"], errors="coerce")
    df["WITHDRAWAL AMT"] = pd.to_numeric(df["WITHDRAWAL AMT"], errors="coerce").fillna(0)
    df["DEPOSIT AMT"] = pd.to_numeric(df["DEPOSIT AMT"], errors="coerce").fillna(0)
    df["BALANCE AMT"] = pd.to_numeric(df["BALANCE AMT"], errors="coerce").fillna(0)
    df = df[(df["WITHDRAWAL AMT"] > 0) | (df["DEPOSIT AMT"] > 0)]
    df.dropna(subset=["TRANSACTION DETAILS"], inplace=True)
    df["CATEGORY"] = df["TRANSACTION DETAILS"].apply(categorize_transaction)
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

def apply_dark(fig, axes):
    fig.patch.set_facecolor("#050810")
    for ax in axes:
        ax.set_facecolor("#0a0e1a")
        for s in ax.spines.values(): s.set_color("#1a1f2e")
        ax.tick_params(colors="#555", labelsize=8)

def crore_fmt(x, pos): return f"₹{x/1e7:.0f}Cr"

st.markdown("""
<div style="padding:36px 0 12px 0;">
  <span style="font-size:52px;font-weight:900;letter-spacing:-3px;">Fin</span><span style="font-size:52px;font-weight:900;letter-spacing:-3px;background:linear-gradient(135deg,#00f5a0,#00d4ff,#7b61ff);-webkit-background-clip:text;-webkit-text-fill-color:transparent;">Sight</span>
  <p style="font-size:11px;color:#444;margin-top:8px;font-family:DM Mono,monospace;letter-spacing:3px;">AI-POWERED FINANCIAL INTELLIGENCE</p>
</div>
<hr style="border:none;border-top:1px solid #1a1f2e;margin-bottom:24px;">
""", unsafe_allow_html=True)

tab1, tab2, tab3, tab4 = st.tabs(["📤  Upload & Analyze","📊  Dashboard","📈  Forecast","🤖  AI Advisor"])

with tab1:
    col1, col2 = st.columns([1,1])
    with col1:
        st.markdown("### Upload Your Bank Statement")
        st.caption("SUPPORTS CSV AND XLSX FORMAT")
        uploaded_file = st.file_uploader("Drop your file here", type=["csv","xlsx"])
        if uploaded_file:
            if st.button("⚡  Analyze My Finances"):
                with st.spinner("Running ML pipeline..."):
                    try:
                        df = process_file(uploaded_file)
                        st.session_state.user_df = df
                        st.session_state.analysis_done = True
                        st.success(f"Done! Analyzed {len(df):,} transactions.")
                    except Exception as e:
                        st.error(f"Error: {e}")
    with col2:
        st.markdown("### Expected Columns")
        st.dataframe(pd.DataFrame({
            "DATE":["2023-01-01"],"TRANSACTION DETAILS":["NEFT/XYZ"],
            "WITHDRAWAL AMT":[5000],"DEPOSIT AMT":[0],"BALANCE AMT":[95000]
        }), use_container_width=True, hide_index=True)

    if st.session_state.analysis_done and st.session_state.user_df is not None:
        df = st.session_state.user_df
        st.markdown("<hr style='border:none;border-top:1px solid #1a1f2e;margin:24px 0'>", unsafe_allow_html=True)
        c1,c2,c3,c4 = st.columns(4)
        with c1: st.markdown(f'<div class="metric-card"><div class="metric-value">{len(df):,}</div><div class="metric-label">Transactions</div></div>', unsafe_allow_html=True)
        with c2: st.markdown(f'<div class="metric-card"><div class="metric-value">₹{df["WITHDRAWAL AMT"].sum()/1e7:.0f}Cr</div><div class="metric-label">Total Withdrawn</div></div>', unsafe_allow_html=True)
        with c3: st.markdown(f'<div class="metric-card"><div class="metric-value">{int(df["IS_ANOMALY"].sum()):,}</div><div class="metric-label">Anomalies Flagged</div></div>', unsafe_allow_html=True)
        with c4:
            cov = f'{(df["CATEGORY"] != "Other").sum() / len(df) * 100:.1f}%'
            st.markdown(f'<div class="metric-card"><div class="metric-value">{cov}</div><div class="metric-label">Auto-Categorized</div></div>', unsafe_allow_html=True)
        st.markdown("### Sample Transactions")
        d = df[["DATE","TRANSACTION DETAILS","WITHDRAWAL AMT","DEPOSIT AMT","CATEGORY","IS_ANOMALY"]].head(25).copy()
        d["IS_ANOMALY"] = d["IS_ANOMALY"].map({0:"Normal",1:"⚠ Anomaly"})
        st.dataframe(d, use_container_width=True, hide_index=True)

with tab2:
    if st.session_state.user_df is None:
        st.warning(" Upload your bank statement in the Upload tab first.")
    else:
        df = st.session_state.user_df
        fig, axes = plt.subplots(2, 2, figsize=(16,10))
        apply_dark(fig, axes.flat)
        fig.suptitle("FinSight — Financial Overview", fontsize=16, fontweight="bold", color="white", y=0.98)
        monthly_wd = df.groupby(df["DATE"].dt.to_period("M"))["WITHDRAWAL AMT"].sum()
        axes[0,0].plot(range(len(monthly_wd)), monthly_wd.values, color="#00f5a0", linewidth=2)
        axes[0,0].fill_between(range(len(monthly_wd)), monthly_wd.values, alpha=0.1, color="#00f5a0")
        axes[0,0].set_title("Monthly Withdrawals", color="#ddd")
        axes[0,0].yaxis.set_major_formatter(mticker.FuncFormatter(crore_fmt))
        step = max(1, len(monthly_wd)//8)
        axes[0,0].set_xticks(range(0,len(monthly_wd),step))
        axes[0,0].set_xticklabels(monthly_wd.index.astype(str)[::step], rotation=45, ha="right", fontsize=7)
        cat_counts = df["CATEGORY"].value_counts()
        colors = ["#00f5a0","#00d4ff","#7b61ff","#ff4d6d","#ffd60a","#ff9f43","#a8e063","#f8a5c2","#778ca3","#2d98da","#4b7bec","#a55eea"]
        axes[0,1].barh(cat_counts.index, cat_counts.values, color=colors[:len(cat_counts)])
        axes[0,1].set_title("Transactions by Category", color="#ddd")
        axes[0,1].invert_yaxis()
        anomaly_monthly = df[df["IS_ANOMALY"]==1].groupby("MONTH")["IS_ANOMALY"].count()
        if len(anomaly_monthly) > 0:
            x = list(range(len(anomaly_monthly)))
            axes[1,0].bar(x, anomaly_monthly.values, color="#ff4d6d", alpha=0.8)
            axes[1,0].set_title("Anomalies Per Month", color="#ddd")
            step2 = max(1,len(x)//8)
            axes[1,0].set_xticks(x[::step2])
            axes[1,0].set_xticklabels(anomaly_monthly.index.astype(str)[::step2], rotation=45, ha="right", fontsize=7)
        wd = df[df["WITHDRAWAL AMT"]>0]["WITHDRAWAL AMT"]
        axes[1,1].hist(wd, bins=50, color="#00d4ff", edgecolor="#050810", log=True)
        axes[1,1].set_title("Withdrawal Distribution (log)", color="#ddd")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        ca, cb = st.columns(2)
        with ca:
            st.markdown("### Spending by Category")
            cs = df.groupby("CATEGORY")["WITHDRAWAL AMT"].sum().sort_values(ascending=False)
            st.dataframe(pd.DataFrame({"Category":cs.index,"Total (₹ Cr)":(cs.values/1e7).round(2)}), use_container_width=True, hide_index=True)
        with cb:
            st.markdown("### Top Anomalous Transactions")
            an = df[df["IS_ANOMALY"]==1].nlargest(10,"WITHDRAWAL AMT")[["DATE","TRANSACTION DETAILS","WITHDRAWAL AMT","ANOMALY_SCORE"]].copy()
            an["WITHDRAWAL AMT"] = (an["WITHDRAWAL AMT"]/1e7).round(3)
            an.columns = ["Date","Transaction","Amount (₹ Cr)","Score"]
            st.dataframe(an, use_container_width=True, hide_index=True)

with tab3:
    if st.session_state.user_df is None:
        st.warning(" Upload your bank statement in the Upload tab first.")
    else:
        df = st.session_state.user_df
        st.markdown("### 6-Month Expense Forecast")
        st.caption("Powered by Facebook Prophet")
        with st.spinner("Training forecast model..."):
            try:
                from prophet import Prophet
                ms = df.groupby(df["DATE"].dt.to_period("M"))["WITHDRAWAL AMT"].sum().reset_index()
                ms.columns = ["ds","y"]
                ms["ds"] = ms["ds"].dt.to_timestamp()
                m = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False, changepoint_prior_scale=0.05)
                m.fit(ms)
                future = m.make_future_dataframe(periods=6, freq="MS")
                fc = m.predict(future)
                fig2, ax2 = plt.subplots(2,1,figsize=(14,10))
                apply_dark(fig2, ax2)
                fig2.suptitle("FinSight — Expense Forecast", fontsize=16, fontweight="bold", color="white")
                ax2[0].plot(ms["ds"], ms["y"], color="#00d4ff", linewidth=2, label="Actual")
                ax2[0].plot(fc["ds"], fc["yhat"], color="#00f5a0", linewidth=2, linestyle="--", label="Forecast")
                ax2[0].fill_between(fc["ds"], fc["yhat_lower"], fc["yhat_upper"], alpha=0.15, color="#00f5a0")
                ax2[0].axvline(x=ms["ds"].max(), color="#ff4d6d", linestyle="--", linewidth=1.5, label="Forecast Start")
                ax2[0].set_title("6-Month Withdrawal Forecast", color="#ddd")
                ax2[0].yaxis.set_major_formatter(mticker.FuncFormatter(crore_fmt))
                ax2[0].legend(facecolor="#0a0e1a", labelcolor="white", fontsize=9)
                fo = fc.tail(6)
                ax2[1].bar(fo["ds"].dt.strftime("%Y-%m"), fo["yhat"], color="#7b61ff", alpha=0.85)
                ax2[1].set_title("Predicted Spending — Next 6 Months", color="#ddd")
                ax2[1].yaxis.set_major_formatter(mticker.FuncFormatter(crore_fmt))
                plt.tight_layout()
                st.pyplot(fig2)
                plt.close()
                ft = fo[["ds","yhat","yhat_lower","yhat_upper"]].copy()
                ft["Month"] = ft["ds"].dt.strftime("%Y-%m")
                ft["Predicted (₹ Cr)"] = (ft["yhat"]/1e7).round(2)
                ft["Lower (₹ Cr)"] = (ft["yhat_lower"]/1e7).round(2)
                ft["Upper (₹ Cr)"] = (ft["yhat_upper"]/1e7).round(2)
                st.dataframe(ft[["Month","Predicted (₹ Cr)","Lower (₹ Cr)","Upper (₹ Cr)"]], use_container_width=True, hide_index=True)
            except Exception as e:
                st.error(f"Forecast error: {e}")

with tab4:
    st.markdown("### AI Financial Advisor")
    st.caption("POWERED BY LLAMA 3.3 70B VIA GROQ")
    if st.session_state.user_df is None:
        st.warning("Upload your bank statement in the Upload tab first.")
    elif not GROQ_API_KEY:
        st.error("Add GROQ_API_KEY in Streamlit Cloud → App Settings → Secrets")
    else:
        df = st.session_state.user_df
        total_wd = df["WITHDRAWAL AMT"].sum()/1e7
        total_dep = df["DEPOSIT AMT"].sum()/1e7
        anomaly_count = int(df["IS_ANOMALY"].sum())
        avg_monthly = df.groupby("MONTH")["WITHDRAWAL AMT"].sum().mean()/1e7
        top_cat = df["CATEGORY"].value_counts().index[0]
        cat_text = "\n".join([f"- {c}: ₹{a/1e7:.1f}Cr" for c,a in df.groupby("CATEGORY")["WITHDRAWAL AMT"].sum().sort_values(ascending=False).items()])
        context = f"""You are FinSight, a personal AI financial advisor.
User bank data:
- Total Withdrawals: ₹{total_wd:.1f} Crores
- Total Deposits: ₹{total_dep:.1f} Crores
- Most Common Transaction: {top_cat}
- Suspicious Transactions: {anomaly_count}
- Avg Monthly Spending: ₹{avg_monthly:.1f} Crores
- Total Transactions: {len(df):,}
- Date Range: {df['DATE'].min().date()} to {df['DATE'].max().date()}
Spending by Category:
{cat_text}
Be concise, friendly, data-backed. Use ₹ Crores."""
        qcols = st.columns(3)
        for i, q in enumerate(["What is my biggest spending category?","How many suspicious transactions do I have?","Give me 3 tips to reduce my spending!"]):
            with qcols[i]:
                if st.button(q, key=f"q{i}"):
                    st.session_state.chat_history.append({"role":"user","content":q})
                    with st.spinner("Thinking..."):
                        r = groq_client.chat.completions.create(model="llama-3.3-70b-versatile", messages=[{"role":"system","content":context},{"role":"user","content":q}])
                        st.session_state.chat_history.append({"role":"assistant","content":r.choices[0].message.content})
                    st.rerun()
        for msg in st.session_state.chat_history:
            css = "chat-user" if msg["role"]=="user" else "chat-ai"
            st.markdown(f'<div class="{css}">{msg["content"]}</div>', unsafe_allow_html=True)
        user_input = st.chat_input("Ask anything about your finances...")
        if user_input:
            st.session_state.chat_history.append({"role":"user","content":user_input})
            with st.spinner("FinSight is thinking..."):
                msgs = [{"role":"system","content":context}] + st.session_state.chat_history[-10:]
                r = groq_client.chat.completions.create(model="llama-3.3-70b-versatile", messages=msgs)
                st.session_state.chat_history.append({"role":"assistant","content":r.choices[0].message.content})
            st.rerun()
        if st.session_state.chat_history:
            if st.button("Clear Chat"):
                st.session_state.chat_history = []
                st.rerun()

st.markdown("<hr style='border:none;border-top:1px solid #1a1f2e;margin-top:40px'><p style='text-align:center;font-size:10px;color:#333;font-family:DM Mono,monospace;letter-spacing:2px;padding-bottom:16px;'>FINSIGHT · AI-POWERED FINANCIAL INTELLIGENCE</p>", unsafe_allow_html=True)
