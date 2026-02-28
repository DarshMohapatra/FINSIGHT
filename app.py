import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import os
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from groq import Groq
try:
    import pdfplumber
    PDF_OK = True
except ImportError:
    PDF_OK = False

st.set_page_config(page_title="FinSight", page_icon="⚡", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Mono:wght@400;500&display=swap');
*{box-sizing:border-box;}
.stApp{background:#020408!important;color:#e8eaf0!important;font-family:'Syne',sans-serif!important;}
footer,#MainMenu,header{display:none!important;}
.block-container{padding:0!important;max-width:100%!important;}
.stButton>button{background:linear-gradient(135deg,#00f5a0,#00d4ff)!important;color:#000!important;font-weight:700!important;border:none!important;border-radius:10px!important;padding:14px 32px!important;font-size:15px!important;width:100%!important;}
.stButton>button:hover{transform:translateY(-2px)!important;box-shadow:0 8px 32px rgba(0,245,160,0.3)!important;}
.stTabs [data-baseweb="tab-list"]{background:transparent!important;border-bottom:1px solid rgba(255,255,255,0.06)!important;padding:0 40px!important;}
.stTabs [data-baseweb="tab"]{background:transparent!important;color:rgba(255,255,255,0.3)!important;font-family:'DM Mono',monospace!important;font-size:11px!important;letter-spacing:2px!important;text-transform:uppercase!important;padding:16px 24px!important;border:none!important;}
.stTabs [aria-selected="true"]{color:#00f5a0!important;border-bottom:2px solid #00f5a0!important;background:transparent!important;}
[data-testid="stFileUploader"]{background:rgba(255,255,255,0.03)!important;border:1px dashed rgba(255,255,255,0.1)!important;border-radius:16px!important;padding:20px!important;}
.stSuccess{background:rgba(0,245,160,0.08)!important;border:1px solid rgba(0,245,160,0.2)!important;border-radius:10px!important;}
.stError{background:rgba(255,60,100,0.08)!important;border:1px solid rgba(255,60,100,0.2)!important;border-radius:10px!important;}
::-webkit-scrollbar{width:4px;}::-webkit-scrollbar-thumb{background:#1a1f2e;border-radius:2px;}
.glass{background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.08);border-radius:20px;padding:28px;margin-bottom:16px;}
.metric-card{background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.08);border-radius:16px;padding:24px;text-align:center;margin-bottom:12px;}
.metric-val{font-size:28px;font-weight:800;background:linear-gradient(135deg,#00f5a0,#00d4ff);-webkit-background-clip:text;-webkit-text-fill-color:transparent;font-family:'DM Mono',monospace!important;}
.metric-lbl{font-size:11px;color:rgba(255,255,255,0.35);font-family:'DM Mono',monospace!important;letter-spacing:1.5px;text-transform:uppercase;margin-top:6px;}
.divider{border:none;border-top:1px solid rgba(255,255,255,0.06);margin:32px 0;}
.chat-u{background:linear-gradient(135deg,#00f5a0,#00d4ff);color:#000;padding:14px 18px;border-radius:18px 18px 4px 18px;margin:8px 0 8px 20%;font-weight:500;font-size:14px;line-height:1.6;}
.chat-ai{background:rgba(255,255,255,0.04);border:1px solid rgba(255,255,255,0.08);color:#e8eaf0;padding:14px 18px;border-radius:18px 18px 18px 4px;margin:8px 20% 8px 0;font-size:14px;line-height:1.6;}
.chat-lbl{font-family:'DM Mono',monospace;font-size:10px;color:rgba(255,255,255,0.25);letter-spacing:1px;margin-bottom:4px;}
.app-wrap{padding:32px 40px;}
</style>
""", unsafe_allow_html=True)

for k, v in {"page": "landing", "user_df": None, "chat_history": [], "analysis_done": False}.items():
    if k not in st.session_state:
        st.session_state[k] = v

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
groq_client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None


def categorize(desc):
    d = str(desc).upper()
    if any(k in d for k in ["SALARY", "SAL ", "PAYROLL", "GIBL"]): return "Salary"
    elif any(k in d for k in ["TRF FROM", "TRANSFER IN", "IMPS/CR", "NEFT/CR"]): return "Transfer In"
    elif any(k in d for k in ["TRF TO", "TRANSFER OUT"]): return "Transfer Out"
    elif any(k in d for k in ["NEFT", "RTGS", "IMPS"]): return "Online Payment"
    elif any(k in d for k in ["CASHDEP", "CASH DEP", "CASH DEPOSIT"]): return "Cash Deposit"
    elif any(k in d for k in ["ATM", "CDM", "CASHWDL", "CASH WDL"]): return "ATM/Cash Withdrawal"
    elif "UPI" in d: return "UPI Payment"
    elif any(k in d for k in ["CHQ", "CHEQUE", "CHECK"]): return "Cheque"
    elif any(k in d for k in ["EMI", "LOAN", "MORTGAGE"]): return "Loan/EMI"
    elif any(k in d for k in ["TAX", "GST", "GOVT", "TDS"]): return "Tax/Government"
    else: return "Other"


def normalize_columns(df):
    df.columns = df.columns.str.strip()
    direct = {
        "Category": "TRANSACTION DETAILS", "Type": "TRANSACTION DETAILS",
        "Narration": "TRANSACTION DETAILS", "Description": "TRANSACTION DETAILS",
        "Particulars": "TRANSACTION DETAILS", "Timestamp": "DATE",
        "Transaction Date": "DATE", "Txn Date": "DATE", "Trans Date": "DATE",
        "Debit": "WITHDRAWAL AMT", "Debit Amt": "WITHDRAWAL AMT",
        "Credit": "DEPOSIT AMT", "Credit Amt": "DEPOSIT AMT",
        "Balance": "BALANCE AMT", "Closing Balance": "BALANCE AMT",
    }
    for src, tgt in direct.items():
        if src in df.columns and tgt not in df.columns:
            df = df.rename(columns={src: tgt})
    cols_upper = {c.upper().strip(): c for c in df.columns}
    date_c   = ["DATE", "TRANSACTION DATE", "TXN DATE", "VALUE DATE", "TIMESTAMP"]
    detail_c = ["TRANSACTION DETAILS", "NARRATION", "DESCRIPTION", "PARTICULARS", "TYPE", "CATEGORY", "REMARKS"]
    wd_c     = ["WITHDRAWAL AMT", "DEBIT", "DEBIT AMT", "DR", "WITHDRAWALS", "AMOUNT DEBITED"]
    dep_c    = ["DEPOSIT AMT", "CREDIT", "CREDIT AMT", "CR", "DEPOSITS", "AMOUNT CREDITED"]
    bal_c    = ["BALANCE AMT", "BALANCE", "CLOSING BALANCE", "RUNNING BALANCE"]

    def find(cands):
        for c in cands:
            if c in cols_upper: return cols_upper[c]
        for c in cands:
            for cu, co in cols_upper.items():
                if c in cu or cu in c: return co
        return None

    col_map = {}
    for tgt, cands in [("DATE", date_c), ("TRANSACTION DETAILS", detail_c),
                       ("WITHDRAWAL AMT", wd_c), ("DEPOSIT AMT", dep_c), ("BALANCE AMT", bal_c)]:
        f = find(cands)
        if f and tgt not in df.columns:
            col_map[f] = tgt
    df = df.rename(columns=col_map)
    missing = [r for r in ["DATE", "TRANSACTION DETAILS", "WITHDRAWAL AMT"] if r not in df.columns]
    if missing:
        raise ValueError(
            "Cannot find required columns: " + str(missing) +
            "\nYour file has: " + str(list(df.columns)) +
            "\nPlease rename to: DATE, TRANSACTION DETAILS, WITHDRAWAL AMT, DEPOSIT AMT, BALANCE AMT"
        )
    for col in ["DEPOSIT AMT", "BALANCE AMT"]:
        if col not in df.columns: df[col] = 0
    return df


def parse_pdf(f, pwd=""):
    import io
    if not PDF_OK:
        raise ValueError("pdfplumber not installed.")
    raw = f.read(); f.seek(0)
    kw = {"password": pwd} if pwd.strip() else {}
    try:
        pdf = pdfplumber.open(io.BytesIO(raw), **kw)
    except Exception as e:
        if any(w in str(e).lower() for w in ["password","encrypt","incorrect"]):
            raise ValueError("Wrong password. Try PAN number, DOB (DDMMYYYY), or account number.")
        raise ValueError(f"Cannot open PDF: {e}")
    tables = []
    with pdf:
        for pg in pdf.pages:
            for t in (pg.extract_tables() or []):
                if t and len(t) > 2:
                    tables.append(t)
    if not tables:
        raise ValueError("No tables found in PDF. Please download the digital (non-scanned) statement from NetBanking, or use CSV/Excel.")
    FIN_KW = ["date","debit","credit","balance","withdrawal","deposit","narration","particulars","description","amount","dr","cr","mode"]
    def find_header_row(tbl):
        for i, r in enumerate(tbl):
            if not r: continue
            cells = [str(c).strip().lower() for c in r if c]
            hits = sum(1 for c in cells if any(k in c for k in FIN_KW))
            if hits >= 2:
                return i
        return 0
    best, best_score = None, 0
    for tbl in tables:
        hi = find_header_row(tbl)
        hdrs = [str(c).strip() if c else f"c{j}" for j,c in enumerate(tbl[hi])]
        rows = tbl[hi+1:]
        if len(rows) < 2: continue
        score = sum(1 for h in hdrs if any(k in h.lower() for k in FIN_KW)) + len(rows)*0.05
        if score > best_score:
            best_score = score
            try: best = (hdrs, rows)
            except: pass
    if not best:
        raise ValueError("Could not find transaction table in PDF. Try CSV/Excel.")
    hdrs, rows = best
    def clean(v):
        if v is None or str(v).strip() in ["","-","None"]: return 0.0
        s = str(v).replace(",","").replace(" ","").strip()
        neg = s.startswith("(") and s.endswith(")")
        s = s.strip("()").replace("Dr","").replace("Cr","")
        try: return (-1 if neg else 1)*float(s)
        except: return 0.0
    def fcol(df, cands):
        up = {c.upper().strip():c for c in df.columns}
        for ca in cands:
            k = ca.upper().strip()
            if k in up: return up[k]
            for u,c in up.items():
                if k in u or u in k: return c
        return None
    try:
        df = pd.DataFrame(rows, columns=hdrs).dropna(how="all")
    except Exception as e:
        raise ValueError(f"Could not parse PDF table: {e}")
    date_c = fcol(df, ["Date","Txn Date","Transaction Date","Tran Date","Value Date","Trans Date"])
    det_c  = fcol(df, ["Narration","Description","Particulars","Transaction Remarks","Details","Transaction Details","Remarks"])
    dr_c   = fcol(df, ["Withdrawal Amt (Dr)","Withdrawal Amt","Withdrawal","Debit","Dr","Dr Amount","Debit Amount"])
    cr_c   = fcol(df, ["Deposit Amt (Cr)","Deposit Amt","Deposit","Credit","Cr","Cr Amount","Credit Amount"])
    bal_c  = fcol(df, ["Balance","Closing Balance","Balance Amt","Running Balance"])
    amt_c  = fcol(df, ["Amount","Transaction Amount"])
    typ_c  = fcol(df, ["Type","Cr/Dr","Dr/Cr","Txn Type","Trans Type"])
    if not date_c:
        raise ValueError(f"Could not find Date column. Found: {list(df.columns)}. Try CSV/Excel.")
    out = pd.DataFrame()
    out["DATE"] = df[date_c]
    out["TRANSACTION DETAILS"] = df[det_c].fillna("").astype(str) if det_c else "TRANSACTION"
    if dr_c and cr_c:
        out["WITHDRAWAL AMT"] = df[dr_c].apply(clean)
        out["DEPOSIT AMT"]    = df[cr_c].apply(clean)
    elif amt_c and typ_c:
        amts = df[amt_c].apply(clean)
        typs = df[typ_c].astype(str).str.upper()
        out["WITHDRAWAL AMT"] = amts.where(typs.str.contains("DR|DEBIT"), 0)
        out["DEPOSIT AMT"]    = amts.where(typs.str.contains("CR|CREDIT"), 0)
    elif amt_c:
        amts = df[amt_c].apply(clean)
        out["WITHDRAWAL AMT"] = amts.apply(lambda x: abs(x) if x<0 else 0)
        out["DEPOSIT AMT"]    = amts.apply(lambda x: x if x>0 else 0)
    else:
        out["WITHDRAWAL AMT"] = 0
        out["DEPOSIT AMT"]    = 0
    out["BALANCE AMT"] = df[bal_c].apply(clean) if bal_c else 0
    bad = ["date","txn date","transaction date","tran date","value date",""]
    out = out[~out["DATE"].astype(str).str.strip().str.lower().isin(bad)]
    if out.empty:
        raise ValueError("PDF parsed but no valid rows found. Try CSV/Excel.")
    return out


def process_file(uploaded, pdf_password=""):
    if uploaded.name.lower().endswith(".pdf"):
        df = parse_pdf(uploaded, pdf_password)
        try: df = normalize_columns(df)
        except: pass
    elif uploaded.name.lower().endswith(".csv"):
        df = pd.read_csv(uploaded)
        df = normalize_columns(df)
    else:
        df = pd.read_excel(uploaded)
        df = normalize_columns(df)
    df["DATE"] = pd.to_datetime(df["DATE"], errors="coerce")
    df["WITHDRAWAL AMT"] = pd.to_numeric(df["WITHDRAWAL AMT"], errors="coerce").fillna(0)
    df["DEPOSIT AMT"]    = pd.to_numeric(df["DEPOSIT AMT"],    errors="coerce").fillna(0)
    df["BALANCE AMT"]    = pd.to_numeric(df["BALANCE AMT"],    errors="coerce").fillna(0)
    df = df[(df["WITHDRAWAL AMT"] > 0) | (df["DEPOSIT AMT"] > 0)].copy()
    df.dropna(subset=["DATE"], inplace=True)
    df["CATEGORY"] = df["TRANSACTION DETAILS"].apply(categorize)
    feats = df[["WITHDRAWAL AMT", "DEPOSIT AMT", "BALANCE AMT"]].copy()
    sc = StandardScaler()
    fs = sc.fit_transform(feats)
    m = IsolationForest(n_estimators=100, contamination=0.02, random_state=42)
    m.fit(fs)
    df["IS_ANOMALY"] = m.predict(fs)
    df["IS_ANOMALY"] = df["IS_ANOMALY"].map({1: 0, -1: 1})
    df["MONTH"] = df["DATE"].dt.to_period("M")
    return df


def dark_fig(rows, cols, size):
    fig, axes = plt.subplots(rows, cols, figsize=size)
    fig.patch.set_facecolor("#020408")
    axs = axes.flat if hasattr(axes, "flat") else [axes]
    for ax in axs:
        ax.set_facecolor("#0a0e1a")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_color("#1a1f2e")
        ax.spines["left"].set_color("#1a1f2e")
        ax.tick_params(colors="#555")
    return fig, axes


def cfmt(x, pos):
    raw = abs(x)
    if raw >= 1e7:   return "Rs." + str(round(x/1e7, 1)) + "Cr"
    elif raw >= 1e5: return "Rs." + str(round(x/1e5, 1)) + "L"
    elif raw >= 1e3: return "Rs." + str(round(x/1e3, 0))[:-2] + "K"
    else:            return "Rs." + str(round(x, 0))[:-2]


def build_context(df):
    raw_total = df["WITHDRAWAL AMT"].sum()
    if raw_total >= 1e7:   scale, slbl = 1e7, "Crores"
    elif raw_total >= 1e5: scale, slbl = 1e5, "Lakhs"
    else:                  scale, slbl = 1,   "Rupees"
    total_wd    = round(df["WITHDRAWAL AMT"].sum() / scale, 2)
    total_dep   = round(df["DEPOSIT AMT"].sum() / scale, 2)
    avg_monthly = round(df.groupby("MONTH")["WITHDRAWAL AMT"].sum().mean() / scale, 2)
    anomalies   = int(df["IS_ANOMALY"].sum())
    top_cat     = df["CATEGORY"].value_counts().index[0]
    monthly_t   = df.groupby("MONTH")["WITHDRAWAL AMT"].sum()
    max_month   = str(monthly_t.idxmax())
    min_month   = str(monthly_t.idxmin())
    top5        = df.nlargest(5, "WITHDRAWAL AMT")
    top5_lines  = []
    for _, row in top5.iterrows():
        top5_lines.append("  - " + row["DATE"].strftime("%d %b %Y") + " | " + str(row["TRANSACTION DETAILS"])[:40] + " | " + cfmt(row["WITHDRAWAL AMT"], None))
    trend_parts = []
    for mo, val in monthly_t.items():
        trend_parts.append(str(mo) + ": " + str(round(val/scale, 2)))
    cat_parts = []
    for c, a in df.groupby("CATEGORY")["WITHDRAWAL AMT"].sum().sort_values(ascending=False).items():
        cat_parts.append("  - " + c + ": " + str(round(a/scale, 2)) + " " + slbl)
    flow = "SURPLUS" if total_dep > total_wd else "DEFICIT"
    ctx = "You are FinSight, a smart personal AI financial advisor.\n"
    ctx += "REAL USER BANK DATA (all amounts in " + slbl + "):\n"
    ctx += "Date Range: " + df["DATE"].min().strftime("%d %b %Y") + " to " + df["DATE"].max().strftime("%d %b %Y") + "\n"
    ctx += "Total Transactions: " + str(len(df)) + "\n"
    ctx += "Total Withdrawals: " + str(total_wd) + " " + slbl + "\n"
    ctx += "Total Deposits: " + str(total_dep) + " " + slbl + "\n"
    ctx += "Net Flow: " + str(round(total_dep-total_wd, 2)) + " " + slbl + " (" + flow + ")\n"
    ctx += "Avg Monthly Spending: " + str(avg_monthly) + " " + slbl + "\n"
    ctx += "Highest Spending Month: " + max_month + "\n"
    ctx += "Lowest Spending Month: " + min_month + "\n"
    ctx += "Anomalies Flagged: " + str(anomalies) + "\n"
    ctx += "Top Category: " + top_cat + "\n"
    ctx += "Monthly Trend:\n" + " | ".join(trend_parts) + "\n"
    ctx += "Top 5 Largest Withdrawals:\n" + "\n".join(top5_lines) + "\n"
    ctx += "Category Breakdown:\n" + "\n".join(cat_parts) + "\n"
    ctx += "IMPORTANT: Use ONLY this data. Always give specific numbers. Never say data unavailable. Be concise, friendly and actionable."
    return ctx

""")

#  HTML landing page
with open('/content/app.py', 'a', encoding='utf-8') as f:
    f.write("""
LANDING_HTML = open('/mount/src/finsight/landing.html').read() if os.path.exists('/mount/src/finsight/landing.html') else ""

if st.session_state.page == "landing":
    components.html(LANDING_HTML, height=2400, scrolling=True)
    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("⚡ Launch App — Analyze My Finances", use_container_width=True):
            st.session_state.page = "app"
            st.rerun()
else:
    st.markdown("<div class='app-wrap'>", unsafe_allow_html=True)
    c1, c2 = st.columns([4, 1])
    with c1:
        st.markdown('<div style="padding:20px 0 12px;"><span style="font-size:28px;font-weight:800;">⚡<span style="background:linear-gradient(135deg,#00f5a0,#00d4ff);-webkit-background-clip:text;-webkit-text-fill-color:transparent;">FinSight</span></span><span style="font-family:DM Mono,monospace;font-size:11px;color:rgba(255,255,255,0.25);margin-left:16px;letter-spacing:2px;">AI FINANCIAL INTELLIGENCE</span></div>', unsafe_allow_html=True)
    with c2:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("← Home"):
            st.session_state.page = "landing"
            st.rerun()
    st.markdown("<hr class='divider'>", unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs(["📤  UPLOAD", "📊  DASHBOARD", "📈  FORECAST", "🤖  AI ADVISOR"])

    with tab1:
        c1, c2 = st.columns([2, 1])
        with c1:
            st.markdown('<div style="font-family:DM Mono,monospace;font-size:11px;color:#00f5a0;letter-spacing:2px;margin:28px 0 8px;">UPLOAD YOUR FILE</div>', unsafe_allow_html=True)
            st.markdown('<h3 style="font-size:24px;font-weight:700;margin-bottom:8px;">Bank Statement Analyzer</h3>', unsafe_allow_html=True)
            st.markdown('<p style="color:rgba(255,255,255,0.4);font-size:14px;margin-bottom:24px;">Upload your bank statement — CSV, Excel, or PDF (password protected supported). Auto-detect &amp; full ML pipeline.</p>', unsafe_allow_html=True)
            uploaded = st.file_uploader("", type=["csv", "xlsx", "xls", "pdf"], label_visibility="collapsed")
            pdf_password = ""
            if uploaded and uploaded.name.lower().endswith(".pdf"):
                st.markdown('<div style="font-family:DM Mono,monospace;font-size:10px;color:rgba(255,255,255,0.4);letter-spacing:1px;margin:10px 0 4px;">🔒 PDF PASSWORD (if protected)</div>', unsafe_allow_html=True)
                pdf_password = st.text_input("", placeholder="PAN number / DOB (DDMMYYYY) / account number", type="password", label_visibility="collapsed")
                st.caption("💡 Leave blank if the PDF has no password")
            if uploaded:
                if st.button("⚡ Run Analysis", use_container_width=True):
                    with st.spinner("Running ML pipeline..."):
                        try:
                            df = process_file(uploaded, pdf_password)
                            st.session_state.user_df = df
                            st.session_state.analysis_done = True
                            st.success("✅ Analysis complete! Found " + str(len(df)) + " transactions.")
                        except Exception as e:
                            st.error("❌ " + str(e))
        with c2:
            st.markdown('<div class="glass" style="margin-top:72px;"><div style="font-family:DM Mono,monospace;font-size:10px;color:#00f5a0;letter-spacing:2px;margin-bottom:16px;">AUTO-DETECTED COLUMNS</div><table style="width:100%;border-collapse:collapse;font-size:12px;"><tr style="color:rgba(255,255,255,0.3);"><td style="padding:6px 0;font-family:DM Mono,monospace;">Required As</td><td style="padding:6px 0;font-family:DM Mono,monospace;">Also Accepted</td></tr><tr style="border-top:1px solid rgba(255,255,255,0.06);"><td style="padding:8px 0;color:#00f5a0;font-family:DM Mono,monospace;font-size:11px;">DATE</td><td style="padding:8px 0;color:rgba(255,255,255,0.4);font-size:11px;">Transaction Date, Timestamp</td></tr><tr style="border-top:1px solid rgba(255,255,255,0.06);"><td style="padding:8px 0;color:#00f5a0;font-family:DM Mono,monospace;font-size:11px;">TRANSACTION DETAILS</td><td style="padding:8px 0;color:rgba(255,255,255,0.4);font-size:11px;">Narration, Type, Category</td></tr><tr style="border-top:1px solid rgba(255,255,255,0.06);"><td style="padding:8px 0;color:#00f5a0;font-family:DM Mono,monospace;font-size:11px;">WITHDRAWAL AMT</td><td style="padding:8px 0;color:rgba(255,255,255,0.4);font-size:11px;">Debit, Debit Amt, DR</td></tr><tr style="border-top:1px solid rgba(255,255,255,0.06);"><td style="padding:8px 0;color:rgba(255,255,255,0.5);font-family:DM Mono,monospace;font-size:11px;">DEPOSIT AMT</td><td style="padding:8px 0;color:rgba(255,255,255,0.4);font-size:11px;">Credit, Credit Amt, CR</td></tr><tr style="border-top:1px solid rgba(255,255,255,0.06);"><td style="padding:8px 0;color:rgba(255,255,255,0.5);font-family:DM Mono,monospace;font-size:11px;">BALANCE AMT</td><td style="padding:8px 0;color:rgba(255,255,255,0.4);font-size:11px;">Balance, Closing Balance</td></tr></table></div>', unsafe_allow_html=True)

        if st.session_state.analysis_done and st.session_state.user_df is not None:
            df = st.session_state.user_df
            st.markdown("<hr class='divider'>", unsafe_allow_html=True)
            st.markdown('<div style="font-family:DM Mono,monospace;font-size:11px;color:#00f5a0;letter-spacing:2px;margin-bottom:20px;">ANALYSIS SUMMARY</div>', unsafe_allow_html=True)
            c1, c2, c3, c4 = st.columns(4)
            for col, (val, lbl) in zip([c1, c2, c3, c4], [
                (str(len(df)), "TRANSACTIONS"),
                (cfmt(df["WITHDRAWAL AMT"].sum(), None), "TOTAL WITHDRAWN"),
                (str(int(df["IS_ANOMALY"].sum())), "ANOMALIES FLAGGED"),
                (str(round((df["CATEGORY"] != "Other").sum()/len(df)*100, 1)) + "%", "AUTO CATEGORIZED"),
            ]):
                with col:
                    st.markdown("<div class='metric-card'><div class='metric-val'>" + val + "</div><div class='metric-lbl'>" + lbl + "</div></div>", unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)
            disp = df[["DATE", "TRANSACTION DETAILS", "WITHDRAWAL AMT", "DEPOSIT AMT", "CATEGORY", "IS_ANOMALY"]].head(20).copy()
            disp["IS_ANOMALY"] = disp["IS_ANOMALY"].map({1: "🚨 Yes", 0: "✅ No"})
            disp["WITHDRAWAL AMT"] = disp["WITHDRAWAL AMT"].apply(lambda x: cfmt(x, None))
            disp["DEPOSIT AMT"]    = disp["DEPOSIT AMT"].apply(lambda x: cfmt(x, None))
            st.dataframe(disp, use_container_width=True, hide_index=True)

    with tab2:
        if st.session_state.user_df is None:
            st.markdown('<div style="text-align:center;padding:100px 0;"><div style="font-size:56px;margin-bottom:16px;">📤</div><div style="font-family:DM Mono,monospace;font-size:12px;color:rgba(255,255,255,0.2);letter-spacing:2px;">UPLOAD A FILE FIRST</div></div>', unsafe_allow_html=True)
        else:
            df = st.session_state.user_df
            st.markdown('<div style="padding:28px 0 0;font-family:DM Mono,monospace;font-size:11px;color:#00f5a0;letter-spacing:2px;margin-bottom:20px;">YOUR FINANCIAL DASHBOARD</div>', unsafe_allow_html=True)
            fig, axes = dark_fig(2, 2, (16, 10))
            fig.suptitle("FinSight Financial Dashboard", fontsize=16, fontweight="bold", color="white", y=0.98)
            pal = ["#00f5a0","#00d4ff","#7b61ff","#ff4d6d","#ffd60a","#ff9f43","#a8e063","#f8a5c2","#778ca3","#2d98da","#4b7bec","#a55eea"]
            mwd = df.groupby(df["DATE"].dt.to_period("M"))["WITHDRAWAL AMT"].sum()
            axes[0,0].plot(range(len(mwd)), mwd.values, color="#00f5a0", linewidth=2.5)
            axes[0,0].fill_between(range(len(mwd)), mwd.values, alpha=0.1, color="#00f5a0")
            axes[0,0].set_title("Monthly Withdrawals", color="white", fontsize=12, pad=12)
            axes[0,0].yaxis.set_major_formatter(mticker.FuncFormatter(cfmt))
            step = max(1, len(mwd)//8)
            axes[0,0].set_xticks(range(0, len(mwd), step))
            axes[0,0].set_xticklabels(mwd.index.astype(str)[::step], rotation=45, ha="right", color="#555", fontsize=8)
            cc = df["CATEGORY"].value_counts()
            axes[0,1].barh(cc.index, cc.values, color=pal[:len(cc)])
            axes[0,1].set_title("Transactions by Category", color="white", fontsize=12, pad=12)
            axes[0,1].tick_params(colors="#aaa", labelsize=9)
            axes[0,1].invert_yaxis()
            am = df[df["IS_ANOMALY"]==1].groupby("MONTH")["IS_ANOMALY"].count()
            x = list(range(len(am)))
            axes[1,0].bar(x, am.values, color="#ff4d6d", alpha=0.85, edgecolor="none")
            axes[1,0].set_title("Anomalies Per Month", color="white", fontsize=12, pad=12)
            step2 = max(1, len(x)//8)
            axes[1,0].set_xticks(x[::step2])
            axes[1,0].set_xticklabels(am.index.astype(str)[::step2], rotation=45, ha="right", color="#555", fontsize=8)
            wd = df[df["WITHDRAWAL AMT"] > 0]["WITHDRAWAL AMT"]
            axes[1,1].hist(wd, bins=50, color="#00d4ff", edgecolor="#020408", log=True, alpha=0.9)
            axes[1,1].set_title("Withdrawal Distribution (log)", color="white", fontsize=12, pad=12)
            axes[1,1].tick_params(colors="#555")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

    with tab3:
        if st.session_state.user_df is None:
            st.markdown('<div style="text-align:center;padding:100px 0;"><div style="font-size:56px;margin-bottom:16px;">📈</div><div style="font-family:DM Mono,monospace;font-size:12px;color:rgba(255,255,255,0.2);letter-spacing:2px;">UPLOAD A FILE FIRST</div></div>', unsafe_allow_html=True)
        else:
            df = st.session_state.user_df
            st.markdown('<div style="padding:28px 0 0;font-family:DM Mono,monospace;font-size:11px;color:#00f5a0;letter-spacing:2px;margin-bottom:8px;">6-MONTH EXPENSE FORECAST</div>', unsafe_allow_html=True)
            with st.spinner("Training forecast model..."):
                try:
                    from prophet import Prophet
                    ms = df.groupby(df["DATE"].dt.to_period("M"))["WITHDRAWAL AMT"].sum().reset_index()
                    ms.columns = ["ds", "y"]
                    ms["ds"] = ms["ds"].dt.to_timestamp()
                    m = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False, changepoint_prior_scale=0.05)
                    m.fit(ms)
                    future = m.make_future_dataframe(periods=6, freq="MS")
                    fc = m.predict(future)
                    fig2, ax2 = dark_fig(2, 1, (14, 10))
                    ax2[0].plot(ms["ds"], ms["y"], color="#00c8e0", linewidth=2.5, label="Actual")
                    ax2[0].plot(fc["ds"], fc["yhat"], color="#00f0a0", linewidth=2, linestyle="--", label="Forecast")
                    ax2[0].fill_between(fc["ds"], fc["yhat_lower"], fc["yhat_upper"], alpha=0.12, color="#00f0a0")
                    ax2[0].axvline(x=ms["ds"].max(), color="#ff3c64", linestyle="--", linewidth=1.5, alpha=0.7, label="Forecast Start")
                    ax2[0].set_title("Monthly Withdrawal Forecast", fontsize=13, pad=14)
                    ax2[0].yaxis.set_major_formatter(mticker.FuncFormatter(cfmt))
                    ax2[0].legend(facecolor="#080d1a", labelcolor="#c8d0e0", fontsize=9)
                    fo = fc.tail(6)
                    bars = ax2[1].bar(fo["ds"].dt.strftime("%b %Y"), fo["yhat"], color="#7b61ff", alpha=0.85, width=0.6)
                    ax2[1].set_title("Predicted Spending Next 6 Months", fontsize=13, pad=14)
                    ax2[1].yaxis.set_major_formatter(mticker.FuncFormatter(cfmt))
                    for bar, val in zip(bars, fo["yhat"]):
                        ax2[1].text(bar.get_x()+bar.get_width()/2, val+max(fo["yhat"])*0.02, cfmt(val, None), ha="center", va="bottom", fontsize=9, color="#c8d0e0")
                    plt.tight_layout(pad=3)
                    st.pyplot(fig2)
                    plt.close()
                    ft = fo[["ds","yhat","yhat_lower","yhat_upper"]].copy()
                    ft["Month"]     = ft["ds"].dt.strftime("%B %Y")
                    ft["Predicted"] = ft["yhat"].apply(lambda x: cfmt(x, None))
                    ft["Lower"]     = ft["yhat_lower"].apply(lambda x: cfmt(x, None))
                    ft["Upper"]     = ft["yhat_upper"].apply(lambda x: cfmt(x, None))
                    st.dataframe(ft[["Month","Predicted","Lower","Upper"]], use_container_width=True, hide_index=True)
                except Exception as e:
                    st.error("Forecast error: " + str(e))
                    st.info("Need at least 6 months of data for reliable forecasting.")

    with tab4:
        st.markdown('<div style="padding:28px 0 0;font-family:DM Mono,monospace;font-size:11px;color:#00f5a0;letter-spacing:2px;margin-bottom:8px;">AI FINANCIAL ADVISOR</div>', unsafe_allow_html=True)
        if st.session_state.user_df is None:
            st.markdown('<div style="text-align:center;padding:100px 0;"><div style="font-size:56px;margin-bottom:16px;">🤖</div><div style="font-family:DM Mono,monospace;font-size:12px;color:rgba(255,255,255,0.2);letter-spacing:2px;">UPLOAD A FILE FIRST</div></div>', unsafe_allow_html=True)
        elif not GROQ_API_KEY:
            st.error("GROQ_API_KEY not set in Streamlit Secrets")
        else:
            df = st.session_state.user_df
            context = build_context(df)
            qc1, qc2, qc3 = st.columns(3)
            quick_qs = [
                "What is my biggest spending category and how can I reduce it?",
                "Are there suspicious transactions I should investigate?",
                "Give me a 3-step plan to improve my finances."
            ]
            for i, (col, q) in enumerate(zip([qc1, qc2, qc3], quick_qs)):
                with col:
                    if st.button(q[:36]+"...", key="qq"+str(i), use_container_width=True):
                        st.session_state.chat_history.append({"role": "user", "content": q})
                        with st.spinner("Thinking..."):
                            r = groq_client.chat.completions.create(
                                model="llama-3.3-70b-versatile",
                                messages=[{"role": "system", "content": context}, {"role": "user", "content": q}],
                                max_tokens=600
                            )
                            st.session_state.chat_history.append({"role": "assistant", "content": r.choices[0].message.content})
                        st.rerun()
            st.markdown("<br>", unsafe_allow_html=True)
            for msg in st.session_state.chat_history:
                if msg["role"] == "user":
                    st.markdown("<div class='chat-lbl' style='text-align:right;'>YOU</div><div class='chat-u'>" + msg["content"] + "</div>", unsafe_allow_html=True)
                else:
                    st.markdown("<div class='chat-lbl'>FINSIGHT AI</div><div class='chat-ai'>" + msg["content"] + "</div>", unsafe_allow_html=True)
            user_input = st.chat_input("Ask anything about your finances...")
            if user_input:
                st.session_state.chat_history.append({"role": "user", "content": user_input})
                with st.spinner("Thinking..."):
                    msgs = [{"role": "system", "content": context}] + st.session_state.chat_history[-12:]
                    r = groq_client.chat.completions.create(model="llama-3.3-70b-versatile", messages=msgs, max_tokens=800)
                    st.session_state.chat_history.append({"role": "assistant", "content": r.choices[0].message.content})
                st.rerun()
            if st.session_state.chat_history:
                if st.button("🗑  Clear Chat"):
                    st.session_state.chat_history = []
                    st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown('<div style="border-top:1px solid rgba(255,255,255,0.06);margin-top:60px;padding:24px 40px;"><span style="font-family:DM Mono,monospace;font-size:10px;color:rgba(255,255,255,0.15);letter-spacing:2px;">FINSIGHT · ISOLATION FOREST + PROPHET + LLAMA 3.3 70B</span></div>', unsafe_allow_html=True)
""")

print("app.py written:", len(open('/content/app.py').readlines()), "lines")

# Landing page as a separate HTML file
with open('/content/landing.html', 'w', encoding='utf-8') as f:
    f.write("""<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<link href="https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
*{margin:0;padding:0;box-sizing:border-box;}
body{background:#020408;color:#e8eaf0;font-family:'Syne',sans-serif;overflow-x:hidden;}
.bg{position:fixed;inset:0;z-index:0;overflow:hidden;pointer-events:none;}
.grid{position:absolute;inset:0;background-image:linear-gradient(rgba(0,245,160,0.03) 1px,transparent 1px),linear-gradient(90deg,rgba(0,245,160,0.03) 1px,transparent 1px);background-size:60px 60px;animation:gridMove 20s linear infinite;}
@keyframes gridMove{0%{transform:translateY(0)}100%{transform:translateY(60px)}}
.orb{position:absolute;border-radius:50%;filter:blur(100px);animation:orbFloat 8s ease-in-out infinite;}
.o1{width:600px;height:600px;background:radial-gradient(circle,rgba(0,245,160,0.12),transparent 70%);top:-200px;left:-200px;}
.o2{width:500px;height:500px;background:radial-gradient(circle,rgba(0,212,255,0.10),transparent 70%);top:20%;right:-150px;animation-delay:-3s;}
.o3{width:400px;height:400px;background:radial-gradient(circle,rgba(123,97,255,0.10),transparent 70%);bottom:10%;left:30%;animation-delay:-6s;}
@keyframes orbFloat{0%,100%{transform:translateY(0)}50%{transform:translateY(-30px)}}
.wrap{max-width:1100px;margin:0 auto;padding:0 48px;position:relative;z-index:1;}
nav{position:fixed;top:0;left:0;right:0;z-index:100;padding:18px 0;background:rgba(2,4,8,0.8);backdrop-filter:blur(20px);border-bottom:1px solid rgba(255,255,255,0.06);}
.nav-inner{max-width:1100px;margin:0 auto;padding:0 48px;display:flex;align-items:center;justify-content:space-between;}
.logo{font-size:20px;font-weight:800;background:linear-gradient(135deg,#00f5a0,#00d4ff);-webkit-background-clip:text;-webkit-text-fill-color:transparent;}
.nav-links{display:flex;gap:28px;align-items:center;}
.nav-links a{color:rgba(255,255,255,0.4);text-decoration:none;font-size:13px;font-family:'DM Mono',monospace;transition:color 0.2s;}
.nav-links a:hover{color:#00f5a0;}
.hero{min-height:100vh;display:flex;align-items:center;padding-top:80px;}
.badge{display:inline-flex;align-items:center;gap:8px;background:rgba(0,245,160,0.08);border:1px solid rgba(0,245,160,0.2);border-radius:100px;padding:7px 16px;margin-bottom:28px;font-family:'DM Mono',monospace;font-size:11px;color:#00f5a0;letter-spacing:1px;}
.dot{width:6px;height:6px;background:#00f5a0;border-radius:50%;animation:pulse 2s infinite;}
@keyframes pulse{0%,100%{opacity:1;transform:scale(1)}50%{opacity:0.5;transform:scale(0.8)}}
h1{font-size:clamp(52px,7vw,88px);font-weight:800;letter-spacing:-3px;line-height:1.0;margin-bottom:20px;}
.grad{background:linear-gradient(135deg,#00f5a0,#00d4ff,#7b61ff);-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-size:200%;animation:gradShift 3s linear infinite;}
@keyframes gradShift{0%{background-position:0% center}100%{background-position:200% center}}
.desc{font-size:18px;color:rgba(232,234,240,0.5);line-height:1.7;max-width:540px;margin-bottom:36px;}
.actions{display:flex;gap:16px;flex-wrap:wrap;margin-bottom:56px;}
.btn{display:inline-flex;align-items:center;gap:10px;padding:16px 32px;border-radius:12px;font-weight:700;font-size:15px;text-decoration:none;transition:all 0.2s;cursor:pointer;font-family:'Syne',sans-serif;border:none;}
.btn-p{background:linear-gradient(135deg,#00f5a0,#00d4ff);color:#000;box-shadow:0 0 40px rgba(0,245,160,0.3);}
.btn-p:hover{transform:translateY(-2px);box-shadow:0 0 60px rgba(0,245,160,0.5);}
.btn-s{color:rgba(232,234,240,0.5);border:1px solid rgba(255,255,255,0.08)!important;background:rgba(255,255,255,0.03);}
.btn-s:hover{border-color:#00d4ff!important;color:#00d4ff;}
.stats{display:flex;gap:40px;flex-wrap:wrap;}
.stat{border-left:1px solid rgba(255,255,255,0.08);padding-left:20px;}
.stat-num{font-size:28px;font-weight:800;background:linear-gradient(135deg,#00f5a0,#00d4ff);-webkit-background-clip:text;-webkit-text-fill-color:transparent;letter-spacing:-1px;}
.stat-lbl{font-size:10px;color:rgba(255,255,255,0.35);font-family:'DM Mono',monospace;margin-top:4px;}
.section{padding:100px 0;}
.sec-label{font-family:'DM Mono',monospace;font-size:11px;color:#00f5a0;letter-spacing:3px;text-transform:uppercase;margin-bottom:14px;}
.sec-title{font-size:clamp(36px,5vw,54px);font-weight:800;letter-spacing:-2px;line-height:1.1;margin-bottom:56px;}
.grid4{display:grid;grid-template-columns:repeat(2,1fr);gap:2px;background:rgba(255,255,255,0.06);border-radius:24px;overflow:hidden;}
.feat{background:#020408;padding:44px;transition:background 0.3s;}
.feat:hover{background:rgba(255,255,255,0.02);}
.ficon{width:48px;height:48px;border-radius:14px;display:flex;align-items:center;justify-content:center;font-size:20px;margin-bottom:18px;}
.ig{background:rgba(0,245,160,0.1);border:1px solid rgba(0,245,160,0.2);}
.ir{background:rgba(255,77,109,0.1);border:1px solid rgba(255,77,109,0.2);}
.ib{background:rgba(0,212,255,0.1);border:1px solid rgba(0,212,255,0.2);}
.ip{background:rgba(123,97,255,0.1);border:1px solid rgba(123,97,255,0.2);}
.fn{font-family:'DM Mono',monospace;font-size:10px;color:rgba(255,255,255,0.25);letter-spacing:2px;margin-bottom:10px;}
.ft{font-size:20px;font-weight:700;margin-bottom:10px;letter-spacing:-0.5px;}
.fd{font-size:14px;color:rgba(232,234,240,0.45);line-height:1.7;}
.tags{display:flex;gap:6px;flex-wrap:wrap;margin-top:16px;}
.tag{font-family:'DM Mono',monospace;font-size:10px;padding:3px 9px;border-radius:100px;border:1px solid rgba(255,255,255,0.08);color:rgba(255,255,255,0.25);}
.steps-wrap{max-width:660px;margin:0 auto;}
.step{display:flex;gap:28px;padding:36px 0;border-bottom:1px solid rgba(255,255,255,0.06);}
.step:last-child{border:none;}
.snum{font-family:'DM Mono',monospace;font-size:48px;font-weight:300;color:rgba(255,255,255,0.05);flex-shrink:0;width:70px;text-align:right;}
.stag{display:inline-block;font-family:'DM Mono',monospace;font-size:10px;color:#00f5a0;letter-spacing:2px;margin-bottom:8px;}
.step h3{font-size:21px;font-weight:700;letter-spacing:-0.5px;margin-bottom:8px;}
.step p{color:rgba(232,234,240,0.45);line-height:1.7;font-size:14px;}
.chat-box{max-width:660px;margin:0 auto;background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.08);border-radius:24px;overflow:hidden;}
.chat-hdr{padding:18px 22px;border-bottom:1px solid rgba(255,255,255,0.06);display:flex;align-items:center;gap:12px;}
.chat-av{width:34px;height:34px;border-radius:10px;background:linear-gradient(135deg,#00f5a0,#00d4ff);display:flex;align-items:center;justify-content:center;font-size:16px;}
.chat-name{font-size:14px;font-weight:600;}
.chat-status{font-size:11px;color:#00f5a0;font-family:'DM Mono',monospace;}
.messages{padding:22px;display:flex;flex-direction:column;gap:14px;}
.msg{max-width:82%;}
.msg.u{align-self:flex-end;}
.msg.ai{align-self:flex-start;}
.bubble{padding:13px 17px;border-radius:16px;font-size:13px;line-height:1.6;}
.msg.u .bubble{background:linear-gradient(135deg,#00f5a0,#00d4ff);color:#000;font-weight:500;border-bottom-right-radius:4px;}
.msg.ai .bubble{background:rgba(255,255,255,0.05);border:1px solid rgba(255,255,255,0.08);border-bottom-left-radius:4px;}
.mt{font-size:10px;color:rgba(255,255,255,0.25);font-family:'DM Mono',monospace;margin-top:5px;padding:0 3px;}
.msg.u .mt{text-align:right;}
.cta-box{background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.08);border-radius:28px;padding:72px;text-align:center;position:relative;overflow:hidden;}
.cta-glow{position:absolute;width:400px;height:400px;background:radial-gradient(circle,rgba(0,245,160,0.07),transparent 70%);top:50%;left:50%;transform:translate(-50%,-50%);pointer-events:none;}
.reveal{opacity:0;transform:translateY(30px);transition:all 0.7s ease;}
.reveal.on{opacity:1;transform:translateY(0);}
@keyframes fadeUp{from{opacity:0;transform:translateY(30px)}to{opacity:1;transform:translateY(0)}}
.anim{animation:fadeUp 0.6s ease both;}
</style>
</head>
<body>
<div class="bg"><div class="grid"></div><div class="orb o1"></div><div class="orb o2"></div><div class="orb o3"></div></div>
<nav><div class="nav-inner">
  <div class="logo">⚡FinSight</div>
  <div class="nav-links"><a href="#features">Features</a><a href="#how">How it works</a><a href="#chat">AI Advisor</a></div>
</div></nav>
<section class="hero"><div class="wrap">
  <div class="badge anim"><div class="dot"></div> POWERED BY LLAMA 3.3 + ISOLATION FOREST</div>
  <h1 class="anim" style="animation-delay:0.1s">Your Money,<br><span class="grad">Understood.</span></h1>
  <p class="desc anim" style="animation-delay:0.2s">Upload your bank statement. FinSight reads every transaction, flags anomalies, forecasts your future, and gives you a personal AI advisor — in seconds.</p>
  <div class="actions anim" style="animation-delay:0.3s">
    <a href="#cta" class="btn btn-p">⚡ Analyze My Finances</a>
    <a href="#features" class="btn btn-s">→ See How It Works</a>
  </div>
  <div class="stats anim" style="animation-delay:0.4s">
    <div class="stat"><div class="stat-num" data-target="113">0</div><div class="stat-lbl">K+ TRANSACTIONS ANALYZED</div></div>
    <div class="stat"><div class="stat-num" data-target="78">0</div><div class="stat-lbl">% AUTO-CATEGORIZED</div></div>
    <div class="stat"><div class="stat-num" data-target="2275">0</div><div class="stat-lbl">ANOMALIES FLAGGED</div></div>
  </div>
</div></section>
<section class="section" id="features"><div class="wrap">
  <div class="sec-label reveal">CAPABILITIES</div>
  <h2 class="sec-title reveal">Four AI engines.<br><span class="grad">One platform.</span></h2>
  <div class="grid4 reveal">
    <div class="feat"><div class="ficon ig">🧠</div><div class="fn">01 / 04</div><div class="ft">NLP Transaction Categorizer</div><div class="fd">Every transaction auto-labeled using keyword-aware NLP. 78.4% coverage out of the box.</div><div class="tags"><span class="tag">NLP</span><span class="tag">KEYWORD MATCHING</span><span class="tag">AUTO-LABEL</span></div></div>
    <div class="feat"><div class="ficon ir">🚨</div><div class="fn">02 / 04</div><div class="ft">Anomaly Detection Engine</div><div class="fd">Isolation Forest ML model scans every transaction and flags suspicious patterns — unusual amounts, timing spikes.</div><div class="tags"><span class="tag">ISOLATION FOREST</span><span class="tag">SKLEARN</span><span class="tag">REAL-TIME</span></div></div>
    <div class="feat"><div class="ficon ib">📈</div><div class="fn">03 / 04</div><div class="ft">Expense Forecasting</div><div class="fd">Facebook Prophet predicts your next 6 months of spending with confidence intervals.</div><div class="tags"><span class="tag">PROPHET</span><span class="tag">TIME SERIES</span><span class="tag">6-MONTH</span></div></div>
    <div class="feat"><div class="ficon ip">🤖</div><div class="fn">04 / 04</div><div class="ft">AI Financial Advisor</div><div class="fd">Powered by LLaMA 3.3 70B. Knows your actual data. Gives personalized, data-backed advice.</div><div class="tags"><span class="tag">LLAMA 3.3</span><span class="tag">GROQ API</span><span class="tag">RAG</span></div></div>
  </div>
</div></section>
<section class="section" id="how"><div class="wrap">
  <div class="sec-label reveal" style="text-align:center">PROCESS</div>
  <h2 class="sec-title reveal" style="text-align:center">From statement<br><span class="grad">to insight.</span></h2>
  <div class="steps-wrap">
    <div class="step reveal"><div class="snum">01</div><div><div class="stag">UPLOAD</div><h3>Drop your bank statement</h3><p>Upload a CSV or Excel file. Smart parser handles any format automatically.</p></div></div>
    <div class="step reveal"><div class="snum">02</div><div><div class="stag">ANALYZE</div><h3>AI reads every transaction</h3><p>NLP categorizer, Isolation Forest, and Prophet all run in parallel on your data.</p></div></div>
    <div class="step reveal"><div class="snum">03</div><div><div class="stag">VISUALIZE</div><h3>Your financial story, visualized</h3><p>Dark dashboard with spending trends, categories, anomaly timeline, and forecast.</p></div></div>
    <div class="step reveal"><div class="snum">04</div><div><div class="stag">CHAT</div><h3>Ask anything about your money</h3><p>FinSight knows your real data and gives honest, personalized answers.</p></div></div>
  </div>
</div></section>
<section class="section" id="chat"><div class="wrap">
  <div class="sec-label reveal" style="text-align:center">AI FINANCIAL ADVISOR</div>
  <h2 class="sec-title reveal" style="text-align:center">Talk to your<br><span class="grad">financial data.</span></h2>
  <div class="chat-box reveal">
    <div class="chat-hdr"><div class="chat-av">💡</div><div><div class="chat-name">FinSight AI</div><div class="chat-status"><span style="display:inline-block;width:6px;height:6px;background:#00f5a0;border-radius:50%;margin-right:6px;animation:pulse 2s infinite;"></span>Analyzing your data</div></div></div>
    <div class="messages">
      <div class="msg u"><div class="bubble">What is my biggest spending category?</div><div class="mt">Just now</div></div>
      <div class="msg ai"><div class="bubble">Your biggest category is <strong>Online Payment</strong> at Rs.11,025Cr — 46% of total. Review recurring NEFT transfers to find savings. 💡</div><div class="mt">FinSight AI</div></div>
      <div class="msg u"><div class="bubble">Give me 3 tips to reduce my spending!</div><div class="mt">Just now</div></div>
      <div class="msg ai"><div class="bubble">Based on your real data:<br><br>1. Monitor Online Payments — Rs.11,025Cr. Review recurring transfers.<br>2. Consolidate Accounts — Rs.4,248Cr in transfers.<br>3. Break down Other — Rs.1,504Cr untracked. Find hidden leaks.</div><div class="mt">FinSight AI</div></div>
    </div>
  </div>
</div></section>
<section class="section" id="cta"><div class="wrap">
  <div class="cta-box reveal">
    <div class="cta-glow"></div>
    <div class="sec-label">GET STARTED FREE</div>
    <h2 class="sec-title" style="margin-bottom:16px;">Your finances deserve<br><span class="grad">real intelligence.</span></h2>
    <p style="font-size:17px;color:rgba(232,234,240,0.45);margin-bottom:40px;line-height:1.7;">Get your complete AI-powered financial analysis in under 60 seconds.</p>
  </div>
</div></section>
<div style="border-top:1px solid rgba(255,255,255,0.06);padding:28px 0;position:relative;z-index:1;">
<div class="wrap" style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:16px;">
  <div style="font-size:18px;font-weight:800;background:linear-gradient(135deg,#00f5a0,#00d4ff);-webkit-background-clip:text;-webkit-text-fill-color:transparent;">⚡FinSight</div>
  <div style="font-family:DM Mono,monospace;font-size:10px;color:rgba(255,255,255,0.2);">ISOLATION FOREST + PROPHET + LLAMA 3.3 70B</div>
  <a href="https://github.com/DarshMohapatra/FINSIGHT" style="font-family:DM Mono,monospace;font-size:11px;color:rgba(255,255,255,0.3);text-decoration:none;" target="_blank">GitHub</a>
</div></div>
<script>
const reveals = document.querySelectorAll(".reveal");
const obs = new IntersectionObserver(entries=>{entries.forEach((e,i)=>{if(e.isIntersecting)setTimeout(()=>e.target.classList.add("on"),i*80);});},{threshold:0.1});
reveals.forEach(r=>obs.observe(r));
function animCount(el,target){let s=0;const step=ts=>{if(!s)s=ts;const p=Math.min((ts-s)/2000,1);const e=1-Math.pow(1-p,3);el.textContent=Math.floor(e*target).toLocaleString();if(p<1)requestAnimationFrame(step);};requestAnimationFrame(step);}
setTimeout(()=>{document.querySelectorAll("[data-target]").forEach(n=>animCount(n,parseInt(n.dataset.target)));},500);
</script>
</body></html>
