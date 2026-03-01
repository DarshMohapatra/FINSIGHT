import streamlit as st
import re
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

# ── SESSION STATE GUARD (Fix 3: prevents blank page on rerun) ──
if "show_app" not in st.session_state:
    st.session_state.show_app = False
if "uploaded_file_data" not in st.session_state:
    st.session_state.uploaded_file_data = None
if "uploaded_file_name" not in st.session_state:
    st.session_state.uploaded_file_name = None
if "df_processed" not in st.session_state:
    st.session_state.df_processed = None


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

    # ── Step 1: case-insensitive direct rename map ─────────────────
    direct_map = {
        # TRANSACTION DETAILS aliases
        "CATEGORY": "TRANSACTION DETAILS", "TYPE": "TRANSACTION DETAILS",
        "NARRATION": "TRANSACTION DETAILS", "DESCRIPTION": "TRANSACTION DETAILS",
        "PARTICULARS": "TRANSACTION DETAILS", "DETAILS": "TRANSACTION DETAILS",
        "REMARKS": "TRANSACTION DETAILS", "CHEQUENO": "TRANSACTION DETAILS",
        "CHEQUE NO": "TRANSACTION DETAILS", "MODE": "TRANSACTION DETAILS",
        # DATE aliases
        "TIMESTAMP": "DATE", "TRANSACTION DATE": "DATE", "TXN DATE": "DATE",
        "TRANS DATE": "DATE", "VALUE DATE": "DATE", "POSTING DATE": "DATE",
        "ENTRY DATE": "DATE",
        # WITHDRAWAL aliases
        "DEBIT": "WITHDRAWAL AMT", "DEBIT AMT": "WITHDRAWAL AMT",
        "DR": "WITHDRAWAL AMT", "WITHDRAWALS": "WITHDRAWAL AMT",
        "AMOUNT DEBITED": "WITHDRAWAL AMT", "WD": "WITHDRAWAL AMT",
        # DEPOSIT aliases
        "CREDIT": "DEPOSIT AMT", "CREDIT AMT": "DEPOSIT AMT",
        "CR": "DEPOSIT AMT", "DEPOSITS": "DEPOSIT AMT",
        "AMOUNT CREDITED": "DEPOSIT AMT",
        # BALANCE aliases
        "BALANCE": "BALANCE AMT", "CLOSING BALANCE": "BALANCE AMT",
        "RUNNING BALANCE": "BALANCE AMT", "BAL": "BALANCE AMT",
    }

    # Apply case-insensitive: compare upper-stripped column names
    new_cols = {}
    already_mapped = set()
    for col in df.columns:
        key = col.upper().strip()
        # Only map if target not already present in df and not already claimed
        if key in direct_map:
            tgt = direct_map[key]
            if tgt not in df.columns and tgt not in already_mapped:
                new_cols[col] = tgt
                already_mapped.add(tgt)
    df = df.rename(columns=new_cols)

    # ── Step 2: fuzzy substring match for remaining columns ─────────
    cols_upper = {c.upper().strip(): c for c in df.columns}
    date_c   = ["DATE", "TRANSACTION DATE", "TXN DATE", "VALUE DATE", "TIMESTAMP", "POSTING DATE"]
    detail_c = ["TRANSACTION DETAILS", "PARTICULARS", "NARRATION", "DESCRIPTION",
                "REMARKS", "DETAILS", "TYPE", "CATEGORY", "MODE", "CHEQUE", "TRANS DETAILS"]
    wd_c     = ["WITHDRAWAL AMT", "WITHDRAWALS", "WITHDRAWAL", "DEBIT", "DEBIT AMT", "DR", "AMOUNT DEBITED", "WD AMT"]
    dep_c    = ["DEPOSIT AMT", "DEPOSITS", "DEPOSIT", "CREDIT", "CREDIT AMT", "CR", "AMOUNT CREDITED"]
    bal_c    = ["BALANCE AMT", "BALANCE", "CLOSING BALANCE", "RUNNING BALANCE", "BAL"]

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

    # ── Step 3: last-resort fallback for TRANSACTION DETAILS ────────
    # If still missing, use whichever unmapped string column has
    # the most unique non-numeric values — almost certainly the narration col
    if "TRANSACTION DETAILS" not in df.columns:
        already_used = {"DATE", "WITHDRAWAL AMT", "DEPOSIT AMT", "BALANCE AMT"}
        candidate_cols = [c for c in df.columns if c not in already_used]
        best_col, best_score = None, 0
        for col in candidate_cols:
            try:
                # Score = number of unique non-numeric values (narrations are unique text)
                vals = df[col].dropna().astype(str)
                non_numeric = vals[~vals.str.match(r"^[\d\.,\s\-]+$")]
                score = non_numeric.nunique()
                if score > best_score:
                    best_score = score
                    best_col = col
            except Exception:
                pass
        if best_col:
            df = df.rename(columns={best_col: "TRANSACTION DETAILS"})

    # ── Final check ──────────────────────────────────────────────────
    # TRANSACTION DETAILS is optional — create empty if missing
    # (pdfplumber sometimes can't extract narration column from PDFs)
    if "TRANSACTION DETAILS" not in df.columns:
        # Last resort: use any remaining unmapped string column
        already_used = {"DATE", "WITHDRAWAL AMT", "DEPOSIT AMT", "BALANCE AMT"}
        candidates = [c for c in df.columns if c not in already_used]
        best_col, best_score = None, 0
        for col in candidates:
            try:
                vals = df[col].dropna().astype(str)
                score = vals[~vals.str.match(r"^[\d.,\s\-]+$")].nunique()
                if score > best_score:
                    best_score, best_col = score, col
            except Exception:
                pass
        if best_col and best_score > 0:
            df = df.rename(columns={best_col: "TRANSACTION DETAILS"})
        else:
            # Create empty column — categorize() handles "" → "Other"
            df["TRANSACTION DETAILS"] = ""

    # Only DATE and WITHDRAWAL AMT are truly required
    missing = [c for c in ["DATE", "WITHDRAWAL AMT"] if c not in df.columns]
    if missing:
        raise ValueError(
            "Cannot find required columns: " + str(missing) +
            "\nYour file has: " + str(list(df.columns)) +
            "\nPlease rename to: DATE, TRANSACTION DETAILS, WITHDRAWAL AMT, DEPOSIT AMT, BALANCE AMT"
        )
    for col in ["DEPOSIT AMT", "BALANCE AMT"]:
        if col not in df.columns:
            df[col] = 0
    return df

def _looks_like_date_header(val):
    """True if a string looks like a date column header."""
    if not isinstance(val, str):
        return False
    v = val.strip().lower()
    return any(k in v for k in [
        "date", "txn date", "transaction date", "value date",
        "posting date", "entry date", "timestamp", "dt"
    ])

def _looks_like_date_value(val):
    """True if a string looks like an actual date value."""
    if not isinstance(val, str):
        return False
    patterns = [
        r"\d{1,2}[\-/]\d{1,2}[\-/]\d{2,4}",
        r"\d{1,2}[\-/ ](?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[\-/ ]\d{2,4}",
        r"\d{4}[\-/]\d{1,2}[\-/]\d{1,2}",
    ]
    return any(re.search(p, val.strip(), re.IGNORECASE) for p in patterns)

def extract_df_from_pdf(pdf_path, password=None):
    import pdfplumber

    open_kwargs = {"password": password} if password else {}
    all_rows = []
    header   = None
    n_cols   = 0

    with pdfplumber.open(pdf_path, **open_kwargs) as pdf:
        for page in pdf.pages:
            # Default extract_tables — this is what originally found 1024 rows
            for tbl in (page.extract_tables() or []):
                if not tbl:
                    continue
                for row in tbl:
                    cells = [str(c or "").strip() for c in row]
                    if not any(cells):
                        continue
                    # Lock header on first row containing a date keyword
                    if header is None:
                        if any(_looks_like_date_header(c) for c in cells):
                            header = cells
                            n_cols = len(cells)
                        continue
                    # Skip repeated header rows
                    if any(_looks_like_date_header(c) for c in cells):
                        continue
                    # Pad/trim and collect
                    while len(cells) < n_cols:
                        cells.append("")
                    all_rows.append(cells[:n_cols])

    if not all_rows or header is None:
        raise ValueError("No transactions found in PDF. Try CSV/Excel.")

    # Build DataFrame with RAW column names from PDF
    df = pd.DataFrame(all_rows, columns=header)

    # Inline rename — map PDF column names to app column names
    # Your PDF has: DATE, MODE**, PARTICULARS, DEPOSITS, WITHDRAWALS, BALANCE
    rename = {}
    for col in df.columns:
        cu = col.upper().strip()
        if "DEPOSIT" in cu and "WITHDRAWAL" not in cu:
            rename[col] = "DEPOSIT AMT"
        elif "WITHDRAWAL" in cu or "WITHDRAW" in cu:
            rename[col] = "WITHDRAWAL AMT"
        elif "BALANCE" in cu:
            rename[col] = "BALANCE AMT"
        elif "PARTICULAR" in cu or "NARRATION" in cu or "DESCRIPTION" in cu or "DETAIL" in cu or "REMARK" in cu:
            rename[col] = "TRANSACTION DETAILS"
        elif _looks_like_date_header(col):
            rename[col] = "DATE"
    df = df.rename(columns=rename)

    df = df.dropna(how="all")
    df = df[df.apply(lambda r: r.astype(str).str.strip().ne("").any(), axis=1)]
    df = df.reset_index(drop=True)
    return df

def process_file(uploaded, pdf_password=""):
    # ── Parse file into raw DataFrame ─────────────────────────────
    if uploaded.name.lower().endswith(".pdf"):
        df = extract_df_from_pdf(uploaded, pdf_password)
        try:
            df = normalize_columns(df)
        except Exception:
            pass
    elif uploaded.name.lower().endswith(".csv"):
        df = pd.read_csv(uploaded)
        df = normalize_columns(df)
    else:
        df = pd.read_excel(uploaded)
        df = normalize_columns(df)

    # ── Parse dates ───────────────────────────────────────────────
    df["DATE"] = pd.to_datetime(df["DATE"], errors="coerce")

    # ── Clean and convert amounts ─────────────────────────────────
    # Bank PDFs use "1,199.00" — commas must be stripped or
    # pd.to_numeric returns NaN, everything becomes 0, pipeline crashes
    for col in ["WITHDRAWAL AMT", "DEPOSIT AMT", "BALANCE AMT"]:
        df[col] = (df[col].astype(str)
                          .str.replace(",", "", regex=False)
                          .str.replace("Rs.", "", regex=False)
                          .str.replace("Rs", "", regex=False)
                          .str.strip())
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    # ── Filter: keep only rows with actual transactions ───────────
    df = df[(df["WITHDRAWAL AMT"] > 0) | (df["DEPOSIT AMT"] > 0)].copy()
    df.dropna(subset=["DATE"], inplace=True)

    if len(df) == 0:
        raise ValueError(
            "No transactions found after filtering. "
            "Check that your WITHDRAWAL AMT and DEPOSIT AMT columns have numeric values."
        )

    # ── NLP Categorization ────────────────────────────────────────
    df["CATEGORY"] = df["TRANSACTION DETAILS"].apply(categorize)

    # ── Anomaly Detection ─────────────────────────────────────────
    feats = df[["WITHDRAWAL AMT", "DEPOSIT AMT", "BALANCE AMT"]].copy()
    sc = StandardScaler()
    fs = sc.fit_transform(feats)
    m  = IsolationForest(n_estimators=100, contamination=0.02, random_state=42)
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


# ── Landing HTML with blank-screen fallback (Fix B) ──────────
_landing_path = '/mount/src/finsight/landing.html'
if os.path.exists(_landing_path):
    LANDING_HTML = open(_landing_path).read()
else:
    # Fallback: show a minimal launch page instead of blank screen
    LANDING_HTML = """<!DOCTYPE html>
<html>
<head>
<style>
  body{margin:0;background:#020408;display:flex;align-items:center;
       justify-content:center;min-height:100vh;font-family:sans-serif;}
  .box{text-align:center;color:#e8eaf0;}
  h1{font-size:48px;background:linear-gradient(135deg,#00f5a0,#00d4ff);
     -webkit-background-clip:text;-webkit-text-fill-color:transparent;}
  p{color:rgba(255,255,255,0.4);margin:12px 0 32px;}
  button{background:linear-gradient(135deg,#00f5a0,#00d4ff);border:none;
         color:#000;font-weight:700;padding:14px 36px;border-radius:10px;
         font-size:15px;cursor:pointer;}
</style>
</head>
<body>
<div class="box">
  <h1>⚡ FinSight</h1>
  <p>AI-powered bank statement analysis</p>
  <button onclick="window.parent.postMessage({type:'finsight_launch'},'*')">
    ⚡ Launch App
  </button>
</div>
</body>
</html>"""


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
            st.markdown('<p style="color:rgba(255,255,255,0.4);font-size:14px;margin-bottom:24px;">Upload any CSV or Excel bank statement. We auto-detect columns and run the full ML pipeline.</p>', unsafe_allow_html=True)
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