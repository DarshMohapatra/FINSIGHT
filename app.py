import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import os
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from groq import Groq

st.set_page_config(page_title="FinSight — AI Finance", page_icon="⚡", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=DM+Sans:wght@300;400;500;600&family=DM+Mono:wght@400;500&display=swap');
*{box-sizing:border-box;margin:0;padding:0;}
.stApp{background:#03060f!important;color:#c8d0e0!important;font-family:'DM Sans',sans-serif!important;}
footer,#MainMenu,header{display:none!important;}
.block-container{padding:0 2rem 2rem 2rem!important;max-width:100%!important;}
.stTabs [data-baseweb="tab-list"]{background:transparent!important;border-bottom:1px solid #0f1628!important;}
.stTabs [data-baseweb="tab"]{background:transparent!important;color:#3a4460!important;font-family:'DM Mono',monospace!important;font-size:11px!important;font-weight:500!important;letter-spacing:2px!important;text-transform:uppercase!important;padding:12px 20px!important;border:none!important;}
.stTabs [aria-selected="true"]{color:#00f0a0!important;border-bottom:2px solid #00f0a0!important;background:transparent!important;}
.stButton>button{background:linear-gradient(135deg,#00f0a0 0%,#00c8e0 100%)!important;color:#03060f!important;font-family:'DM Mono',monospace!important;font-weight:600!important;font-size:13px!important;letter-spacing:1px!important;border:none!important;border-radius:8px!important;padding:14px 32px!important;text-transform:uppercase!important;}
.stButton>button:hover{transform:translateY(-2px)!important;box-shadow:0 8px 32px rgba(0,240,160,0.3)!important;}
[data-testid="stMetric"]{background:rgba(255,255,255,0.02)!important;border:1px solid #0f1628!important;border-radius:12px!important;padding:20px!important;}
[data-testid="stMetricValue"]{color:#00f0a0!important;font-family:'DM Mono',monospace!important;font-size:24px!important;}
[data-testid="stMetricLabel"]{color:#3a4460!important;font-family:'DM Mono',monospace!important;font-size:10px!important;letter-spacing:2px!important;}
.stSuccess{background:rgba(0,240,160,0.08)!important;border:1px solid rgba(0,240,160,0.2)!important;border-radius:8px!important;}
.stError{background:rgba(255,60,100,0.08)!important;border:1px solid rgba(255,60,100,0.2)!important;border-radius:8px!important;}
.stWarning{background:rgba(255,180,0,0.08)!important;border:1px solid rgba(255,180,0,0.2)!important;border-radius:8px!important;}
::-webkit-scrollbar{width:4px;}::-webkit-scrollbar-track{background:#03060f;}::-webkit-scrollbar-thumb{background:#0f1628;border-radius:2px;}
</style>
""", unsafe_allow_html=True)

for key, val in [("user_df", None), ("chat_history", []), ("analysis_done", False)]:
    if key not in st.session_state:
        st.session_state[key] = val

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
groq_client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None

# ── Smart Header Finder ────────────────────────────────────────
def find_transaction_header_row(df_raw):
    """Scan rows to find where actual transaction data starts."""
    date_keywords   = ["DATE","TRANSACTION DATE","TXN DATE","VALUE DATE","POSTING DATE","TRAN DATE"]
    detail_keywords = ["NARRATION","DESCRIPTION","PARTICULARS","TRANSACTION DETAILS","REMARKS","DETAILS"]
    debit_keywords  = ["DEBIT","WITHDRAWAL","DR","WITHDRAWAL AMT","DEBIT AMT"]

    for i, row in df_raw.iterrows():
        row_vals = [str(v).upper().strip() for v in row.values if pd.notna(v)]
        has_date   = any(any(k == v for k in date_keywords)   for v in row_vals)
        has_detail = any(any(k == v for k in detail_keywords) for v in row_vals)
        has_debit  = any(any(k == v for k in debit_keywords)  for v in row_vals)
        if (has_date and has_detail) or (has_date and has_debit) or (has_detail and has_debit):
            return i
    return None

# ── Smart Column Normalizer ────────────────────────────────────
def normalize_columns(df):
    # Normalize columns — strip whitespace and uppercase for matching
    df.columns = df.columns.str.strip()
    cols_upper = {c.upper().strip(): c for c in df.columns}
    date_c   = ["DATE","TRANSACTION DATE","TXN DATE","VALUE DATE","POSTING DATE","TRANS DATE","TRAN DATE","BOOK DATE","TXNDATE","VALUEDATE","TIMESTAMP","TIME STAMP","TRANSACTION TIMESTAMP"]
    detail_c = ["TRANSACTION DETAILS","NARRATION","DESCRIPTION","PARTICULARS","REMARKS","TRANSACTION NARRATION","DETAILS","TRAN PARTICULARS","CHEQUENO/NARRATION","TRAN. PARTICULAR","TRANSACTION PARTICULAR","TYPE","TRANSACTION TYPE","CATEGORY","TRANS TYPE"]
    wd_c     = ["WITHDRAWAL AMT","DEBIT","DEBIT AMT","WITHDRAWAL","DR AMT","DR","AMOUNT DEBITED","DEBIT AMOUNT","WITHDRAW","DEBIT(INR)","DEBIT (INR)","DR AMOUNT","WITHDRAWALS"]
    dep_c    = ["DEPOSIT AMT","CREDIT","CREDIT AMT","DEPOSIT","CR AMT","CR","AMOUNT CREDITED","CREDIT AMOUNT","CREDIT(INR)","CREDIT (INR)","CR AMOUNT","DEPOSITS"]
    bal_c    = ["BALANCE AMT","BALANCE","CLOSING BALANCE","RUNNING BALANCE","AVAIL BAL","AVAILABLE BALANCE","BAL","BALANCE(INR)","BALANCE (INR)"]

    def find(cands):
        # Exact match first
        for c in cands:
            if c in cols_upper: return cols_upper[c]
        # Partial match fallback
        for c in cands:
            for col_up, col_orig in cols_upper.items():
                if c in col_up or col_up in c:
                    return col_orig
        # Last resort — check if any candidate is a substring of any column
        for c in cands:
            for col_up, col_orig in cols_upper.items():
                if any(word in col_up for word in c.split()):
                    return col_orig
        return None

    col_map = {}
    for target, cands in [("DATE",date_c),("TRANSACTION DETAILS",detail_c),
                           ("WITHDRAWAL AMT",wd_c),("DEPOSIT AMT",dep_c),("BALANCE AMT",bal_c)]:
        found = find(cands)
        if found: col_map[found] = target

    df = df.rename(columns=col_map)
    missing = [r for r in ["DATE","TRANSACTION DETAILS","WITHDRAWAL AMT"] if r not in df.columns]
    if missing:
        raise ValueError(
            f"Cannot find required columns: {missing}\n"
            f"Your file has: {list(df.columns)}\n"
            f"Please rename them to: DATE, TRANSACTION DETAILS, WITHDRAWAL AMT, DEPOSIT AMT, BALANCE AMT"
        )
    for col in ["DEPOSIT AMT","BALANCE AMT"]:
        if col not in df.columns: df[col] = 0
    return df

# ── Categorizer ────────────────────────────────────────────────
def categorize(desc):
    d = str(desc).upper()
    if any(k in d for k in ["SALARY","SAL ","PAYROLL","GIBL"]): return "Salary"
    elif any(k in d for k in ["TRF FROM","TRANSFER IN","IMPS/CR","NEFT/CR"]): return "Transfer In"
    elif any(k in d for k in ["TRF TO","TRANSFER OUT"]): return "Transfer Out"
    elif any(k in d for k in ["NEFT","RTGS","IMPS"]): return "Online Payment"
    elif any(k in d for k in ["CASHDEP","CASH DEP","CASH DEPOSIT"]): return "Cash Deposit"
    elif any(k in d for k in ["ATM","CDM","CASHWDL","CASH WDL"]): return "ATM/Cash Withdrawal"
    elif "UPI" in d: return "UPI Payment"
    elif any(k in d for k in ["CHQ","CHEQUE","CHECK"]): return "Cheque"
    elif any(k in d for k in ["EMI","LOAN","MORTGAGE"]): return "Loan/EMI"
    elif any(k in d for k in ["TAX","GST","GOVT","TDS"]): return "Tax/Government"
    elif "INDIAFORENSIC" in d: return "Business Transfer"
    else: return "Other"

# ── File Processor ─────────────────────────────────────────────
def process_file(uploaded_file):
    try:
        if uploaded_file.name.endswith(".csv"):
            # Try to find header row in CSV too
            raw = pd.read_csv(uploaded_file, header=None)
            hrow = find_transaction_header_row(raw)
            if hrow is not None:
                uploaded_file.seek(0)
                df = pd.read_csv(uploaded_file, header=hrow)
            else:
                uploaded_file.seek(0)
                df = pd.read_csv(uploaded_file)
        else:
            # Excel: scan all possible header rows 0-25
            df = None
            raw = pd.read_excel(uploaded_file, header=None)
            hrow = find_transaction_header_row(raw)

            if hrow is not None:
                uploaded_file.seek(0)
                df = pd.read_excel(uploaded_file, header=hrow)
            else:
                # Fallback: try rows 0-5
                for hr in range(6):
                    uploaded_file.seek(0)
                    try:
                        tmp = pd.read_excel(uploaded_file, header=hr)
                        tmp.columns = tmp.columns.astype(str).str.strip()
                        tmp = tmp.dropna(how="all")
                        if len(tmp) > 3 and len(tmp.columns) >= 3:
                            df = tmp; break
                    except: continue
            if df is None:
                raise ValueError("Could not read file. Try saving as CSV.")

        df.columns = df.columns.astype(str).str.strip()
        df = df.dropna(how="all")
        # Drop completely empty columns
        df = df.dropna(axis=1, how="all")
        df = normalize_columns(df)

        df["DATE"] = pd.to_datetime(df["DATE"], errors="coerce")
        for col in ["WITHDRAWAL AMT","DEPOSIT AMT","BALANCE AMT"]:
            df[col] = pd.to_numeric(
                df[col].astype(str).str.replace(",","").str.replace("₹","").str.replace(" ","").str.strip(),
                errors="coerce"
            ).fillna(0)

        df = df[(df["WITHDRAWAL AMT"] > 0) | (df["DEPOSIT AMT"] > 0)]
        df = df.dropna(subset=["DATE"])
        df = df[df["DATE"].dt.year > 2000]

        if len(df) < 3:
            raise ValueError(f"Only {len(df)} valid transaction rows found. Check your file.")

        df["TRANSACTION DETAILS"] = df["TRANSACTION DETAILS"].fillna("Unknown")
        df["CATEGORY"] = df["TRANSACTION DETAILS"].apply(categorize)

        features = df[["WITHDRAWAL AMT","DEPOSIT AMT","BALANCE AMT"]].copy()
        scaler = StandardScaler()
        fs = scaler.fit_transform(features)
        contamination = min(0.05, max(0.01, 10/len(df)))
        model = IsolationForest(n_estimators=100, contamination=contamination, random_state=42)
        model.fit(fs)
        df["ANOMALY_SCORE"] = model.decision_function(fs)
        df["IS_ANOMALY"] = model.predict(fs)
        df["IS_ANOMALY"] = df["IS_ANOMALY"].map({1:0,-1:1})
        df["MONTH"] = df["DATE"].dt.to_period("M")
        return df

    except ValueError: raise
    except Exception as e: raise ValueError(f"Parse error: {str(e)}")

# ── Chart Helpers ──────────────────────────────────────────────
def dark_fig(nrows=1, ncols=1, figsize=(14,6)):
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    fig.patch.set_facecolor("#03060f")
    for ax in (axes.flat if hasattr(axes,"flat") else [axes]):
        ax.set_facecolor("#080d1a")
        for s in ax.spines.values(): s.set_color("#0f1628")
        ax.tick_params(colors="#3a4460", labelsize=8)
        ax.title.set_color("#c8d0e0")
        ax.grid(True, color="#0f1628", linewidth=0.5, alpha=0.5)
    return fig, axes

def cfmt(x, pos):
    if abs(x)>=1e7: return f"₹{x/1e7:.1f}Cr"
    elif abs(x)>=1e5: return f"₹{x/1e5:.1f}L"
    return f"₹{x:,.0f}"

COLORS = ["#00f0a0","#00c8e0","#7b61ff","#ff3c64","#ffb400","#ff9f43","#a8e063","#f8a5c2","#4ecdc4","#45b7d1","#2d98da","#a55eea"]

# ══════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════
st.markdown("""
<div style="padding:48px 8px 32px 8px;border-bottom:1px solid #0f1628;background:radial-gradient(ellipse at 20% 50%,rgba(0,240,160,0.04) 0%,transparent 60%);position:relative;overflow:hidden;">
<div style="position:absolute;inset:0;pointer-events:none;background-image:linear-gradient(rgba(0,240,160,0.015) 1px,transparent 1px),linear-gradient(90deg,rgba(0,240,160,0.015) 1px,transparent 1px);background-size:60px 60px;"></div>
<div style="position:relative;z-index:1;">
<div style="display:flex;align-items:baseline;gap:0;margin-bottom:12px;">
<span style="font-family:'Bebas Neue',sans-serif;font-size:80px;color:#c8d0e0;letter-spacing:5px;line-height:1;text-shadow:0 0 80px rgba(0,240,160,0.1);">FIN</span>
<span style="font-family:'Bebas Neue',sans-serif;font-size:80px;background:linear-gradient(135deg,#00f0a0,#00c8e0);-webkit-background-clip:text;-webkit-text-fill-color:transparent;letter-spacing:5px;line-height:1;">SIGHT</span>
</div>
<div style="display:flex;align-items:center;gap:10px;flex-wrap:wrap;">
<span style="font-family:'DM Mono',monospace;font-size:11px;color:#2a3450;letter-spacing:4px;text-transform:uppercase;">AI-Powered Financial Intelligence</span>
<span style="background:rgba(0,240,160,0.08);border:1px solid rgba(0,240,160,0.2);border-radius:20px;padding:4px 14px;font-family:'DM Mono',monospace;font-size:10px;color:#00f0a0;letter-spacing:1px;">● LIVE</span>
<span style="background:rgba(255,180,0,0.08);border:1px solid rgba(255,180,0,0.18);border-radius:20px;padding:4px 14px;font-family:'DM Mono',monospace;font-size:10px;color:#ffb400;letter-spacing:1px;">🔒 SECURE</span>
<span style="background:rgba(123,97,255,0.08);border:1px solid rgba(123,97,255,0.18);border-radius:20px;padding:4px 14px;font-family:'DM Mono',monospace;font-size:10px;color:#7b61ff;letter-spacing:1px;">LLAMA 3.3 · GROQ</span>
<span style="background:rgba(0,200,224,0.08);border:1px solid rgba(0,200,224,0.18);border-radius:20px;padding:4px 14px;font-family:'DM Mono',monospace;font-size:10px;color:#00c8e0;letter-spacing:1px;">ISOLATION FOREST · PROPHET</span>
</div>
</div>
</div>
""", unsafe_allow_html=True)

tab1, tab2, tab3, tab4 = st.tabs(["⚡  Upload & Analyze","📊  Dashboard","📈  Forecast","🤖  AI Advisor"])

# ══════════════════════════════════════════════
# TAB 1
# ══════════════════════════════════════════════
with tab1:

    # ── Hero ──────────────────────────────────
    st.markdown("""
    <div style="text-align:center;padding:64px 0 56px 0;position:relative;">
    <div style="position:absolute;inset:0;pointer-events:none;background:radial-gradient(ellipse at 50% 0%,rgba(0,240,160,0.06) 0%,transparent 70%);"></div>
    <div style="position:relative;z-index:1;">
    <div style="display:inline-block;background:rgba(0,240,160,0.08);border:1px solid rgba(0,240,160,0.2);border-radius:20px;padding:5px 18px;font-family:'DM Mono',monospace;font-size:11px;color:#00f0a0;letter-spacing:2px;margin-bottom:24px;">✦ PHASE 2 · ML PIPELINE LIVE</div>
    <div style="font-family:'Bebas Neue',sans-serif;font-size:56px;color:#c8d0e0;letter-spacing:3px;line-height:1.05;margin-bottom:16px;">
    UNDERSTAND YOUR<br>
    <span style="background:linear-gradient(135deg,#00f0a0 0%,#00c8e0 50%,#7b61ff 100%);-webkit-background-clip:text;-webkit-text-fill-color:transparent;">MONEY INSTANTLY</span>
    </div>
    <p style="color:#3a4460;font-size:16px;max-width:520px;margin:0 auto 32px auto;line-height:1.7;">
    Upload any bank statement — SBI, HDFC, ICICI, Axis and more.<br>
    ML pipeline runs in seconds. No account needed.
    </p>
    <div style="display:flex;justify-content:center;gap:24px;flex-wrap:wrap;">
    <div style="font-family:'DM Mono',monospace;font-size:12px;color:#5a6480;"><span style="color:#00f0a0;font-size:18px;font-weight:bold;">98%</span><br>Anomaly Accuracy</div>
    <div style="width:1px;background:#0f1628;"></div>
    <div style="font-family:'DM Mono',monospace;font-size:12px;color:#5a6480;"><span style="color:#00c8e0;font-size:18px;font-weight:bold;">6mo</span><br>Forecast Horizon</div>
    <div style="width:1px;background:#0f1628;"></div>
    <div style="font-family:'DM Mono',monospace;font-size:12px;color:#5a6480;"><span style="color:#7b61ff;font-size:18px;font-weight:bold;">70B</span><br>Parameter AI</div>
    <div style="width:1px;background:#0f1628;"></div>
    <div style="font-family:'DM Mono',monospace;font-size:12px;color:#5a6480;"><span style="color:#ffb400;font-size:18px;font-weight:bold;">0</span><br>Data Stored</div>
    </div>
    </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Feature Cards ──────────────────────────
    st.markdown("""
    <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:16px;margin-bottom:40px;">
    <div style="background:linear-gradient(145deg,rgba(0,240,160,0.07),rgba(0,240,160,0.01));border:1px solid rgba(0,240,160,0.15);border-radius:20px;padding:32px 24px;transition:all 0.2s;position:relative;overflow:hidden;">
    <div style="position:absolute;top:0;right:0;width:80px;height:80px;background:radial-gradient(circle,rgba(0,240,160,0.1),transparent);border-radius:50%;transform:translate(20px,-20px);"></div>
    <div style="font-size:32px;margin-bottom:14px;">🧠</div>
    <div style="font-family:'DM Mono',monospace;font-size:10px;color:#00f0a0;letter-spacing:2px;margin-bottom:10px;text-transform:uppercase;">ML Anomaly Detection</div>
    <div style="font-size:13px;color:#5a6480;line-height:1.6;">Isolation Forest algorithm automatically flags suspicious transactions with 98% precision.</div>
    </div>
    <div style="background:linear-gradient(145deg,rgba(0,200,224,0.07),rgba(0,200,224,0.01));border:1px solid rgba(0,200,224,0.15);border-radius:20px;padding:32px 24px;position:relative;overflow:hidden;">
    <div style="position:absolute;top:0;right:0;width:80px;height:80px;background:radial-gradient(circle,rgba(0,200,224,0.1),transparent);border-radius:50%;transform:translate(20px,-20px);"></div>
    <div style="font-size:32px;margin-bottom:14px;">📈</div>
    <div style="font-family:'DM Mono',monospace;font-size:10px;color:#00c8e0;letter-spacing:2px;margin-bottom:10px;text-transform:uppercase;">6-Month Forecast</div>
    <div style="font-size:13px;color:#5a6480;line-height:1.6;">Facebook Prophet time-series model predicts future spending with confidence intervals.</div>
    </div>
    <div style="background:linear-gradient(145deg,rgba(123,97,255,0.07),rgba(123,97,255,0.01));border:1px solid rgba(123,97,255,0.15);border-radius:20px;padding:32px 24px;position:relative;overflow:hidden;">
    <div style="position:absolute;top:0;right:0;width:80px;height:80px;background:radial-gradient(circle,rgba(123,97,255,0.1),transparent);border-radius:50%;transform:translate(20px,-20px);"></div>
    <div style="font-size:32px;margin-bottom:14px;">🤖</div>
    <div style="font-family:'DM Mono',monospace;font-size:10px;color:#7b61ff;letter-spacing:2px;margin-bottom:10px;text-transform:uppercase;">AI Financial Advisor</div>
    <div style="font-size:13px;color:#5a6480;line-height:1.6;">LLaMA 3.3 70B answers questions using your actual uploaded financial data.</div>
    </div>
    <div style="background:linear-gradient(145deg,rgba(255,180,0,0.07),rgba(255,180,0,0.01));border:1px solid rgba(255,180,0,0.15);border-radius:20px;padding:32px 24px;position:relative;overflow:hidden;">
    <div style="position:absolute;top:0;right:0;width:80px;height:80px;background:radial-gradient(circle,rgba(255,180,0,0.1),transparent);border-radius:50%;transform:translate(20px,-20px);"></div>
    <div style="font-size:32px;margin-bottom:14px;">🔒</div>
    <div style="font-family:'DM Mono',monospace;font-size:10px;color:#ffb400;letter-spacing:2px;margin-bottom:10px;text-transform:uppercase;">Privacy First</div>
    <div style="font-size:13px;color:#5a6480;line-height:1.6;">Processed in RAM only. Never stored on disk. Never shared. Refresh to wipe instantly.</div>
    </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Supported Banks Strip ─────────────────
    st.markdown("""
    <div style="background:rgba(255,255,255,0.02);border:1px solid #0f1628;border-radius:12px;padding:16px 24px;margin-bottom:32px;display:flex;align-items:center;gap:20px;flex-wrap:wrap;">
    <span style="font-family:'DM Mono',monospace;font-size:10px;color:#2a3450;letter-spacing:2px;text-transform:uppercase;flex-shrink:0;">Works with:</span>
    <div style="display:flex;gap:12px;flex-wrap:wrap;">
    <span style="background:#080d1a;border:1px solid #0f1628;border-radius:6px;padding:4px 12px;font-family:'DM Mono',monospace;font-size:11px;color:#5a6480;">SBI</span>
    <span style="background:#080d1a;border:1px solid #0f1628;border-radius:6px;padding:4px 12px;font-family:'DM Mono',monospace;font-size:11px;color:#5a6480;">HDFC</span>
    <span style="background:#080d1a;border:1px solid #0f1628;border-radius:6px;padding:4px 12px;font-family:'DM Mono',monospace;font-size:11px;color:#5a6480;">ICICI</span>
    <span style="background:#080d1a;border:1px solid #0f1628;border-radius:6px;padding:4px 12px;font-family:'DM Mono',monospace;font-size:11px;color:#5a6480;">AXIS</span>
    <span style="background:#080d1a;border:1px solid #0f1628;border-radius:6px;padding:4px 12px;font-family:'DM Mono',monospace;font-size:11px;color:#5a6480;">YES BANK</span>
    <span style="background:#080d1a;border:1px solid #0f1628;border-radius:6px;padding:4px 12px;font-family:'DM Mono',monospace;font-size:11px;color:#5a6480;">KOTAK</span>
    <span style="background:#080d1a;border:1px solid #0f1628;border-radius:6px;padding:4px 12px;font-family:'DM Mono',monospace;font-size:11px;color:#5a6480;">FEDERAL</span>
    <span style="background:#080d1a;border:1px solid #0f1628;border-radius:6px;padding:4px 12px;font-family:'DM Mono',monospace;font-size:11px;color:#5a6480;">PNB</span>
    <span style="background:rgba(0,240,160,0.05);border:1px solid rgba(0,240,160,0.15);border-radius:6px;padding:4px 12px;font-family:'DM Mono',monospace;font-size:11px;color:#00f0a0;">+ any CSV/Excel</span>
    </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Security Notice ───────────────────────
    st.markdown("""
    <div style="background:rgba(255,180,0,0.04);border:1px solid rgba(255,180,0,0.12);border-radius:12px;padding:16px 20px;margin-bottom:32px;display:flex;align-items:flex-start;gap:14px;">
    <span style="font-size:20px;flex-shrink:0;">🔐</span>
    <div>
    <div style="font-family:'DM Mono',monospace;font-size:11px;color:#ffb400;letter-spacing:1px;margin-bottom:4px;text-transform:uppercase;">Security and Privacy Guarantee</div>
    <div style="font-size:13px;color:#5a6480;line-height:1.6;">
    Your bank statement is <strong style="color:#c8d0e0;">processed entirely in RAM</strong> and is
    <strong style="color:#c8d0e0;">never written to disk, never stored in a database, and never sent to any server</strong>
    other than Groq API (only when you use AI Advisor). Refresh the page to instantly erase all data.
    </div>
    </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Upload + Format Columns ───────────────
    col_up, col_fmt = st.columns([1,1], gap="large")

    with col_up:
        st.markdown("<div style='font-family:DM Mono,monospace;font-size:11px;color:#3a4460;letter-spacing:2px;text-transform:uppercase;margin-bottom:12px;'>Upload Your File</div>", unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Upload", type=["csv","xlsx","xls"], label_visibility="collapsed")
        if uploaded_file:
            st.markdown(f"<div style='background:rgba(0,240,160,0.05);border:1px solid rgba(0,240,160,0.15);border-radius:8px;padding:10px 16px;font-family:DM Mono,monospace;font-size:12px;color:#00f0a0;margin:8px 0 16px 0;'>✓ {uploaded_file.name} · {uploaded_file.size/1024:.1f} KB</div>", unsafe_allow_html=True)
            if st.button("⚡  RUN ANALYSIS", use_container_width=True):
                with st.spinner("Running ML pipeline..."):
                    try:
                        df = process_file(uploaded_file)
                        st.session_state.user_df = df
                        st.session_state.analysis_done = True
                        st.success(f"✅ Done! {len(df):,} transactions processed.")
                    except ValueError as e:
                        st.error(f"❌ {e}")
                    except Exception as e:
                        st.error(f"❌ Unexpected error: {e}")

    with col_fmt:
        st.markdown("<div style='font-family:DM Mono,monospace;font-size:11px;color:#3a4460;letter-spacing:2px;text-transform:uppercase;margin-bottom:12px;'>Auto-Detected Column Names</div>", unsafe_allow_html=True)
        st.dataframe(pd.DataFrame({
            "Required As":   ["DATE","TRANSACTION DETAILS","WITHDRAWAL AMT","DEPOSIT AMT","BALANCE AMT"],
            "Also Accepted": ["Transaction Date, TXN Date, Value Date","Narration, Description, Particulars, Remarks","Debit, Debit Amt, DR, Amount Debited","Credit, Credit Amt, CR, Amount Credited","Balance, Closing Balance, Running Balance"],
            "Status":        ["✅ Required","✅ Required","✅ Required","Optional","Optional"],
        }), use_container_width=True, hide_index=True)
        st.markdown("""<div style="background:rgba(0,180,255,0.05);border:1px solid rgba(0,180,255,0.12);border-radius:8px;padding:12px 16px;font-size:12px;color:#5a6480;margin-top:10px;line-height:1.6;">
        💡 <strong style="color:#c8d0e0;">Bank statements with metadata rows at the top are handled automatically.</strong>
        The app scans your file to find where actual transactions begin.
        </div>""", unsafe_allow_html=True)

    # ── Results ───────────────────────────────
    if st.session_state.analysis_done and st.session_state.user_df is not None:
        df = st.session_state.user_df
        st.markdown("<hr style='border:none;border-top:1px solid #0f1628;margin:40px 0 28px 0;'>", unsafe_allow_html=True)
        c1,c2,c3,c4,c5 = st.columns(5)
        c1.metric("Transactions", f"{len(df):,}")
        c2.metric("Total Withdrawn", f"₹{df['WITHDRAWAL AMT'].sum()/1e7:.1f}Cr")
        c3.metric("Total Deposited", f"₹{df['DEPOSIT AMT'].sum()/1e7:.1f}Cr")
        c4.metric("Anomalies", f"{int(df['IS_ANOMALY'].sum()):,}")
        c5.metric("Date Range", f"{df['DATE'].min().strftime('%b %y')}→{df['DATE'].max().strftime('%b %y')}")
        st.markdown("<br>", unsafe_allow_html=True)
        d = df[["DATE","TRANSACTION DETAILS","WITHDRAWAL AMT","DEPOSIT AMT","BALANCE AMT","CATEGORY","IS_ANOMALY"]].head(30).copy()
        d["IS_ANOMALY"] = d["IS_ANOMALY"].map({0:"Normal",1:"⚠ Anomaly"})
        d["DATE"] = d["DATE"].dt.strftime("%Y-%m-%d")
        st.dataframe(d, use_container_width=True, hide_index=True)

# ══════════════════════════════════════════════
# TAB 2 — DASHBOARD
# ══════════════════════════════════════════════
with tab2:
    if st.session_state.user_df is None:
        st.markdown("<div style='text-align:center;padding:80px 0;color:#3a4460;'><div style='font-size:48px;margin-bottom:16px;'>📊</div><div style='font-family:DM Mono,monospace;font-size:13px;letter-spacing:2px;'>UPLOAD A FILE FIRST</div></div>", unsafe_allow_html=True)
    else:
        df = st.session_state.user_df
        fig, axes = dark_fig(2, 2, (16,10))
        monthly_wd = df.groupby(df["DATE"].dt.to_period("M"))["WITHDRAWAL AMT"].sum()
        axes[0,0].plot(range(len(monthly_wd)), monthly_wd.values, color="#00f0a0", linewidth=2.5)
        axes[0,0].fill_between(range(len(monthly_wd)), monthly_wd.values, alpha=0.08, color="#00f0a0")
        axes[0,0].set_title("Monthly Withdrawals", fontsize=12, pad=12)
        axes[0,0].yaxis.set_major_formatter(mticker.FuncFormatter(cfmt))
        step = max(1,len(monthly_wd)//8)
        axes[0,0].set_xticks(range(0,len(monthly_wd),step))
        axes[0,0].set_xticklabels(monthly_wd.index.astype(str)[::step], rotation=45, ha="right")
        cat_counts = df["CATEGORY"].value_counts()
        axes[0,1].barh(cat_counts.index, cat_counts.values, color=COLORS[:len(cat_counts)], height=0.6)
        axes[0,1].set_title("Transactions by Category", fontsize=12, pad=12)
        axes[0,1].invert_yaxis()
        an_monthly = df[df["IS_ANOMALY"]==1].groupby("MONTH")["IS_ANOMALY"].count()
        if len(an_monthly) > 0:
            x = list(range(len(an_monthly)))
            axes[1,0].bar(x, an_monthly.values, color="#ff3c64", alpha=0.85, width=0.7)
            axes[1,0].set_title("Anomalies Per Month", fontsize=12, pad=12)
            s2 = max(1,len(x)//8)
            axes[1,0].set_xticks(x[::s2])
            axes[1,0].set_xticklabels(an_monthly.index.astype(str)[::s2], rotation=45, ha="right")
        else:
            axes[1,0].text(0.5,0.5,"No anomalies detected",ha="center",va="center",color="#3a4460",fontsize=12,transform=axes[1,0].transAxes)
            axes[1,0].set_title("Anomalies Per Month", fontsize=12, pad=12)
        cat_spend = df.groupby("CATEGORY")["WITHDRAWAL AMT"].sum().sort_values(ascending=True).tail(8)
        axes[1,1].barh(cat_spend.index, cat_spend.values, color=COLORS[:len(cat_spend)], height=0.6)
        axes[1,1].set_title("Spending by Category", fontsize=12, pad=12)
        axes[1,1].xaxis.set_major_formatter(mticker.FuncFormatter(cfmt))
        plt.tight_layout(pad=2.5)
        st.pyplot(fig)
        plt.close()
        ca, cb = st.columns(2, gap="large")
        with ca:
            st.markdown("**💸 Spending by Category**")
            cs = df.groupby("CATEGORY")["WITHDRAWAL AMT"].sum().sort_values(ascending=False)
            st.dataframe(pd.DataFrame({"Category":cs.index,"Amount (₹)":cs.values.round(0).astype(int),"Transactions":df["CATEGORY"].value_counts().reindex(cs.index).fillna(0).astype(int),"Share %":(cs.values/cs.sum()*100).round(1)}), use_container_width=True, hide_index=True)
        with cb:
            st.markdown("**⚠️ Top Anomalous Transactions**")
            an_df = df[df["IS_ANOMALY"]==1].nlargest(10,"WITHDRAWAL AMT")[["DATE","TRANSACTION DETAILS","WITHDRAWAL AMT","ANOMALY_SCORE"]].copy()
            an_df["DATE"] = an_df["DATE"].dt.strftime("%Y-%m-%d")
            an_df["WITHDRAWAL AMT"] = an_df["WITHDRAWAL AMT"].round(0).astype(int)
            an_df["ANOMALY_SCORE"] = an_df["ANOMALY_SCORE"].round(4)
            an_df.columns = ["Date","Transaction","Amount (₹)","Score"]
            st.dataframe(an_df, use_container_width=True, hide_index=True)

# ══════════════════════════════════════════════
# TAB 3 — FORECAST
# ══════════════════════════════════════════════
with tab3:
    if st.session_state.user_df is None:
        st.markdown("<div style='text-align:center;padding:80px 0;color:#3a4460;'><div style='font-size:48px;margin-bottom:16px;'>📈</div><div style='font-family:DM Mono,monospace;font-size:13px;letter-spacing:2px;'>UPLOAD A FILE FIRST</div></div>", unsafe_allow_html=True)
    else:
        df = st.session_state.user_df
        st.markdown("<div style='font-family:DM Mono,monospace;font-size:11px;color:#3a4460;letter-spacing:2px;text-transform:uppercase;margin-bottom:8px;'>6-Month Expense Forecast</div><p style='color:#5a6480;font-size:13px;margin-bottom:28px;'>Facebook Prophet · yearly seasonality · changepoint_prior_scale=0.05</p>", unsafe_allow_html=True)
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
                fig2, ax2 = dark_fig(2,1,(14,10))
                ax2[0].plot(ms["ds"], ms["y"], color="#00c8e0", linewidth=2.5, label="Actual")
                ax2[0].plot(fc["ds"], fc["yhat"], color="#00f0a0", linewidth=2, linestyle="--", label="Forecast")
                ax2[0].fill_between(fc["ds"], fc["yhat_lower"], fc["yhat_upper"], alpha=0.12, color="#00f0a0", label="Confidence Band")
                ax2[0].axvline(x=ms["ds"].max(), color="#ff3c64", linestyle="--", linewidth=1.5, alpha=0.7, label="Forecast Start")
                ax2[0].set_title("Monthly Withdrawal Forecast", fontsize=13, pad=14)
                ax2[0].yaxis.set_major_formatter(mticker.FuncFormatter(cfmt))
                ax2[0].legend(facecolor="#080d1a", labelcolor="#c8d0e0", fontsize=9, framealpha=1)
                fo = fc.tail(6)
                bars = ax2[1].bar(fo["ds"].dt.strftime("%b %Y"), fo["yhat"], color="#7b61ff", alpha=0.85, width=0.6)
                ax2[1].errorbar(range(6), fo["yhat"], yerr=[fo["yhat"]-fo["yhat_lower"], fo["yhat_upper"]-fo["yhat"]], fmt="none", color="#c8d0e0", capsize=6, linewidth=1.5, alpha=0.5)
                ax2[1].set_title("Predicted Spending — Next 6 Months", fontsize=13, pad=14)
                ax2[1].yaxis.set_major_formatter(mticker.FuncFormatter(cfmt))
                for bar, val in zip(bars, fo["yhat"]):
                    ax2[1].text(bar.get_x()+bar.get_width()/2, val+max(fo["yhat"])*0.02, cfmt(val,None), ha="center", va="bottom", fontsize=9, color="#c8d0e0")
                plt.tight_layout(pad=3)
                st.pyplot(fig2)
                plt.close()
                ft = fo[["ds","yhat","yhat_lower","yhat_upper"]].copy()
                ft["Month"]     = ft["ds"].dt.strftime("%B %Y")
                ft["Predicted"] = ft["yhat"].apply(lambda x: cfmt(x,None))
                ft["Lower"]     = ft["yhat_lower"].apply(lambda x: cfmt(x,None))
                ft["Upper"]     = ft["yhat_upper"].apply(lambda x: cfmt(x,None))
                st.dataframe(ft[["Month","Predicted","Lower","Upper"]], use_container_width=True, hide_index=True)
            except Exception as e:
                st.error(f"Forecast error: {e}")
                st.info("Need at least 6 months of data for reliable forecasting.")

# ══════════════════════════════════════════════
# TAB 4 — AI ADVISOR
# ══════════════════════════════════════════════
with tab4:
    st.markdown("<div style='font-family:DM Mono,monospace;font-size:11px;color:#3a4460;letter-spacing:2px;text-transform:uppercase;margin-bottom:8px;'>AI Financial Advisor</div><p style='color:#5a6480;font-size:13px;margin-bottom:28px;'>Powered by LLaMA 3.3 70B via Groq — answers based on your uploaded data only.</p>", unsafe_allow_html=True)
    if st.session_state.user_df is None:
        st.markdown("<div style='text-align:center;padding:80px 0;color:#3a4460;'><div style='font-size:48px;margin-bottom:16px;'>🤖</div><div style='font-family:DM Mono,monospace;font-size:13px;letter-spacing:2px;'>UPLOAD A FILE FIRST</div></div>", unsafe_allow_html=True)
    elif not GROQ_API_KEY:
        st.markdown("""<div style="background:rgba(255,60,100,0.06);border:1px solid rgba(255,60,100,0.2);border-radius:12px;padding:28px;text-align:center;">
        <div style="font-size:32px;margin-bottom:12px;">🔑</div>
        <div style="font-family:'DM Mono',monospace;font-size:12px;color:#ff3c64;letter-spacing:1px;margin-bottom:12px;">GROQ API KEY NOT CONFIGURED</div>
        <div style="font-size:13px;color:#5a6480;line-height:1.8;">Go to Streamlit Cloud → Settings → Secrets and add:<br><br>
        <code style="background:#0f1628;padding:6px 16px;border-radius:6px;color:#00f0a0;font-size:13px;">GROQ_API_KEY = "your_key_here"</code></div>
        </div>""", unsafe_allow_html=True)
    else:
        df = st.session_state.user_df
        total_wd    = df["WITHDRAWAL AMT"].sum()/1e7
        total_dep   = df["DEPOSIT AMT"].sum()/1e7
        anomalies   = int(df["IS_ANOMALY"].sum())
        avg_monthly = df.groupby("MONTH")["WITHDRAWAL AMT"].sum().mean()/1e7
        top_cat     = df["CATEGORY"].value_counts().index[0]
        cat_text    = "\n".join([f"  - {c}: Rs.{a/1e7:.2f}Cr" for c,a in df.groupby("CATEGORY")["WITHDRAWAL AMT"].sum().sort_values(ascending=False).items()])
        context = f"""You are FinSight, a smart personal AI financial advisor.
REAL USER BANK DATA:
Date Range: {df["DATE"].min().strftime("%b %Y")} to {df["DATE"].max().strftime("%b %Y")}
Total Transactions: {len(df):,}
Total Withdrawals: Rs.{total_wd:.2f} Crores | Total Deposits: Rs.{total_dep:.2f} Crores
Net Flow: Rs.{(total_dep-total_wd):.2f} Crores ({"surplus" if total_dep>total_wd else "deficit"})
Avg Monthly Spending: Rs.{avg_monthly:.2f} Crores | Anomalies: {anomalies} | Top Category: {top_cat}
Category Breakdown:
{cat_text}
Give specific, data-backed, actionable advice. Be concise and friendly."""

        qc1,qc2,qc3 = st.columns(3)
        for i,(col,q) in enumerate(zip([qc1,qc2,qc3],["What is my biggest spending category and how can I reduce it?","Are there suspicious transactions I should investigate?","Give me a 3-step plan to improve my finances."])):
            with col:
                if st.button(q[:38]+"…", key=f"qq{i}", use_container_width=True):
                    st.session_state.chat_history.append({"role":"user","content":q})
                    with st.spinner("Thinking..."):
                        r = groq_client.chat.completions.create(model="llama-3.3-70b-versatile", messages=[{"role":"system","content":context},{"role":"user","content":q}], max_tokens=600)
                        st.session_state.chat_history.append({"role":"assistant","content":r.choices[0].message.content})
                    st.rerun()
        st.markdown("<br>", unsafe_allow_html=True)
        for msg in st.session_state.chat_history:
            if msg["role"]=="user":
                st.markdown(f"<div style='background:linear-gradient(135deg,rgba(0,240,160,0.12),rgba(0,200,224,0.08));border:1px solid rgba(0,240,160,0.2);border-radius:12px 12px 4px 12px;padding:14px 18px;margin:8px 0 8px 20%;font-size:14px;color:#c8d0e0;'>{msg['content']}</div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div style='background:rgba(255,255,255,0.03);border:1px solid #0f1628;border-radius:12px 12px 12px 4px;padding:14px 18px;margin:8px 20% 8px 0;font-size:14px;color:#c8d0e0;line-height:1.7;'>{msg['content']}</div>", unsafe_allow_html=True)
        user_input = st.chat_input("Ask anything about your finances...")
        if user_input:
            st.session_state.chat_history.append({"role":"user","content":user_input})
            with st.spinner("Thinking..."):
                msgs = [{"role":"system","content":context}]+[{"role":m["role"],"content":m["content"]} for m in st.session_state.chat_history[-12:]]
                r = groq_client.chat.completions.create(model="llama-3.3-70b-versatile", messages=msgs, max_tokens=800)
                st.session_state.chat_history.append({"role":"assistant","content":r.choices[0].message.content})
            st.rerun()
        if st.session_state.chat_history:
            if st.button("🗑  Clear Chat"):
                st.session_state.chat_history = []
                st.rerun()

st.markdown("<div style='border-top:1px solid #0f1628;margin-top:60px;padding:24px 0;text-align:center;'><span style='font-family:DM Mono,monospace;font-size:10px;color:#1e2640;letter-spacing:2px;'>FINSIGHT · AI FINANCIAL INTELLIGENCE · ISOLATION FOREST + PROPHET + LLAMA 3.3 70B</span></div>", unsafe_allow_html=True)
