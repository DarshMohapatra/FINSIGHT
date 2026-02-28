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
    # Stage 0: strip whitespace from all column names
    df.columns = [str(c).strip() for c in df.columns]

    # Stage 1: exhaustive exact-name rename map
    direct = {
        "Date": "DATE", "date": "DATE", "DATE": "DATE",
        "Dt": "DATE", "DT": "DATE",
        "Trans Date": "DATE", "Trans. Date": "DATE",
        "Transaction Date": "DATE", "Txn Date": "DATE", "TXN DATE": "DATE",
        "Value Date": "DATE", "VALUE DATE": "DATE",
        "Posting Date": "DATE", "Post Date": "DATE", "Timestamp": "DATE",
        "Narration": "TRANSACTION DETAILS", "NARRATION": "TRANSACTION DETAILS",
        "Particulars": "TRANSACTION DETAILS", "PARTICULARS": "TRANSACTION DETAILS",
        "Description": "TRANSACTION DETAILS", "DESCRIPTION": "TRANSACTION DETAILS",
        "Remarks": "TRANSACTION DETAILS", "Details": "TRANSACTION DETAILS",
        "Type": "TRANSACTION DETAILS", "Category": "TRANSACTION DETAILS",
        "Debit": "WITHDRAWAL AMT", "DEBIT": "WITHDRAWAL AMT",
        "Debit Amt": "WITHDRAWAL AMT", "Debit Amount": "WITHDRAWAL AMT",
        "DR": "WITHDRAWAL AMT", "Dr": "WITHDRAWAL AMT",
        "Withdrawals": "WITHDRAWAL AMT", "WITHDRAWALS": "WITHDRAWAL AMT",
        "Withdrawal": "WITHDRAWAL AMT", "Amount Debited": "WITHDRAWAL AMT",
        "Credit": "DEPOSIT AMT", "CREDIT": "DEPOSIT AMT",
        "Credit Amt": "DEPOSIT AMT", "Credit Amount": "DEPOSIT AMT",
        "CR": "DEPOSIT AMT", "Cr": "DEPOSIT AMT",
        "Deposits": "DEPOSIT AMT", "DEPOSITS": "DEPOSIT AMT",
        "Deposit": "DEPOSIT AMT", "Amount Credited": "DEPOSIT AMT",
        "Balance": "BALANCE AMT", "BALANCE": "BALANCE AMT",
        "Closing Balance": "BALANCE AMT", "CLOSING BALANCE": "BALANCE AMT",
        "Running Balance": "BALANCE AMT", "Bal": "BALANCE AMT", "BAL": "BALANCE AMT",
    }
    for src, tgt in direct.items():
        if src in df.columns and tgt not in df.columns:
            df = df.rename(columns={src: tgt})

    # Stage 2: fully lowercase fuzzy match
    date_c   = ['date', 'transaction date', 'txn date', 'value date',
                'timestamp', 'posting date', 'trans date', 'dt',
                'trans. date', 'post date']
    detail_c = ['transaction details', 'narration', 'description',
                'particulars', 'type', 'category', 'remarks', 'details']
    wd_c     = ['withdrawal amt', 'debit', 'debit amt', 'dr', 'withdrawals',
                'amount debited', 'withdrawal', 'debit amount']
    dep_c    = ['deposit amt', 'credit', 'credit amt', 'cr', 'deposits',
                'amount credited', 'deposit', 'credit amount']
    bal_c    = ['balance amt', 'balance', 'closing balance',
                'running balance', 'bal', 'closing bal']

    cols_lower = {c.lower().strip(): c for c in df.columns}

    def find(cands):
        for c in cands:
            if c in cols_lower:
                return cols_lower[c]
        for c in cands:
            for cl, co in cols_lower.items():
                if c in cl or cl in c:
                    return co
        return None

    col_map = {}
    for tgt, cands in [
        ("DATE",                date_c),
        ("TRANSACTION DETAILS", detail_c),
        ("WITHDRAWAL AMT",      wd_c),
        ("DEPOSIT AMT",         dep_c),
        ("BALANCE AMT",         bal_c),
    ]:
        if tgt not in df.columns:
            f = find(cands)
            if f:
                col_map[f] = tgt
    df = df.rename(columns=col_map)

    # Stage 3: value-scan fallback — if DATE still missing, find it by cell content
    if 'DATE' not in df.columns:
        for col in df.columns:
            sample = df[col].dropna().astype(str).head(20)
            hits   = sample.apply(_looks_like_date_value).sum()
            if hits >= 3:
                df = df.rename(columns={col: 'DATE'})
                break

    # Validate
    missing = [r for r in ["DATE", "TRANSACTION DETAILS", "WITHDRAWAL AMT"]
               if r not in df.columns]
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
    """Return True if a string looks like a date column header."""
    if not isinstance(val, str):
        return False
    val_clean = val.strip().lower()
    date_keywords = [
        "date", "txn date", "transaction date", "value date",
        "posting date", "entry date", "timestamp", "dt"
    ]
    return any(kw in val_clean for kw in date_keywords)

def _looks_like_date_value(val):
    """Return True if val looks like a date. Covers all major Indian bank formats."""
    if val is None:
        return False
    s = str(val).strip()
    # Covers ISO, ICICI (DD-MM-YYYY), slash variants, and month-name formats
    patterns = [
        r'\d{4}-\d{2}-\d{2}',
        r'\d{2}-\d{2}-\d{4}',
        r'\d{2}/\d{2}/\d{4}',
        r'\d{2}/\d{2}/\d{2}',
        r'\d{2}-[A-Za-z]{3}-\d{4}',
        r'\d{1,2}\s+[A-Za-z]{3}\s+\d{4}',
    ]
    return any(re.search(p, s) for p in patterns)


def _score_table(rows):
    """Score a table — higher = more likely to be the transactions table."""
    if not rows or len(rows) < 3:
        return -1
    score = 0

    # Reward: many rows (transaction tables are long)
    score += min(len(rows), 100)  # cap at 100 to avoid dominating

    # Reward: many columns (transaction tables have 4-6 columns)
    avg_cols = sum(len(r) for r in rows) / len(rows)
    if 3 <= avg_cols <= 7:
        score += 30

    # Big reward: header row contains date-like keywords
    header = [str(c or '').strip().lower() for c in rows[0]]
    date_kws   = ['date', 'dt', 'txn', 'trans', 'value date', 'posting']
    detail_kws = ['narration', 'particular', 'description', 'details', 'remarks']
    amount_kws = ['debit', 'credit', 'withdrawal', 'deposit', 'amount', 'dr', 'cr']

    has_date   = any(any(kw in h for kw in date_kws)   for h in header)
    has_detail = any(any(kw in h for kw in detail_kws) for h in header)
    has_amount = any(any(kw in h for kw in amount_kws) for h in header)

    if has_date:   score += 200  # date header is the strongest signal
    if has_detail: score += 100
    if has_amount: score += 100

    # Big penalty: first cell looks like a cover-page / summary label
    first_cell = str(rows[0][0] or '').strip().lower()
    cover_page_signals = [
        'relationship manager', 'customer', 'account holder',
        'statement summary', 'branch', 'address', 'ifsc',
        'dear', 'your', 'balance summary', 'account no',
    ]
    if any(sig in first_cell for sig in cover_page_signals):
        score -= 500  # this is definitely not the transactions table

    # Reward: second row looks like it has date values
    if len(rows) > 1:
        import re as _re
        date_pat = _re.compile(
            r'(\d{2}-\d{2}-\d{4}|\d{4}-\d{2}-\d{2}|\d{2}/\d{2}/\d{4})'
        )
        first_data_row = [str(c or '') for c in rows[1]]
        if any(date_pat.search(cell) for cell in first_data_row):
            score += 300  # data rows with dates = almost certainly right table

    return score



def extract_df_from_pdf(pdf_path, password=None):
    """
    Extract transaction table from a bank statement PDF.
    Handles ICICI, HDFC, Axis, SBI multi-page statements.
    """
    import pdfplumber
    import pandas as pd
    import re as _re

    # Matches DD-MM-YYYY, YYYY-MM-DD, DD/MM/YYYY, DD/MM/YY
    DATE_RE = _re.compile(
        r'(\d{2}-\d{2}-\d{4}|\d{4}-\d{2}-\d{2}|\d{2}/\d{2}/\d{4}|\d{2}/\d{2}/\d{2})'
    )

    def has_date_value(table):
        """Return True if any cell in first 5 rows matches a date pattern."""
        for row in table[:5]:
            for cell in row:
                if cell and DATE_RE.search(str(cell)):
                    return True
        return False

    open_kwargs = {"password": password} if password else {}

    header      = None
    n_cols      = None
    all_rows    = []
    locked      = False   # True once we've found the transactions table

    with pdfplumber.open(pdf_path, **open_kwargs) as pdf:
        for page in pdf.pages:
            for tbl in (page.extract_tables() or []):
                if not tbl or len(tbl) < 2:
                    continue

                if not locked:
                    # Skip tables without date values — they're cover/summary tables
                    if not has_date_value(tbl):
                        continue

                    # Found the transactions table — lock onto it
                    locked = True
                    n_cols = max(len(r) for r in tbl)

                    # Determine header: first row without a date value = header row
                    start = 0
                    first_row = [str(c or '').strip() for c in tbl[0]]
                    if not any(DATE_RE.search(c) for c in first_row):
                        header = first_row
                        start  = 1
                    else:
                        header = [f'COL_{i}' for i in range(n_cols)]

                    for row in tbl[start:]:
                        cells = [str(c or '').strip() for c in row]
                        if any(cells):
                            all_rows.append(cells)

                else:
                    # Already locked — collect rows from matching tables on later pages
                    # Skip if column count is wildly different (it's a different table)
                    tbl_cols = max(len(r) for r in tbl)
                    if abs(tbl_cols - n_cols) > 2:
                        continue

                    # Skip repeated header rows
                    start = 0
                    first_row = [str(c or '').strip() for c in tbl[0]]
                    if not any(DATE_RE.search(c) for c in first_row):
                        start = 1  # no dates = header row, skip it

                    for row in tbl[start:]:
                        cells = [str(c or '').strip() for c in row]
                        if any(cells):
                            all_rows.append(cells)

    if not locked or not all_rows:
        raise ValueError(
            'No transaction table found in this PDF. '
            'Make sure it is a bank statement with a Date/Amount table.'
        )

    # Pad/trim all rows to same column count
    padded = []
    for row in all_rows:
        if len(row) < n_cols:
            row = row + [''] * (n_cols - len(row))
        padded.append(row[:n_cols])

    if len(header) < n_cols:
        header = header + [f'COL_{i}' for i in range(len(header), n_cols)]
    header = header[:n_cols]

    df = pd.DataFrame(padded, columns=header)

    # Drop fully empty rows
    df = df.replace('', float('nan'))
    df = df.dropna(how='all')
    df = df.fillna('')
    df = df.reset_index(drop=True)

    # Remove repeated header rows (ICICI prints header on every page)
    first_col = df.columns[0]
    df = df[df[first_col].astype(str).str.strip() != str(first_col).strip()]
    df = df.reset_index(drop=True)

    df = normalize_columns(df)
    return df


def process_file(uploaded, pdf_password=""):
    if uploaded.name.lower().endswith(".pdf"):

        import pdfplumber as _plumber, io as _io
        _raw = uploaded.read()
        uploaded.seek(0)
        _pw = {"password": pdf_password} if pdf_password else {}
        with _plumber.open(_io.BytesIO(_raw), **_pw) as _pdf:
            st.write("Total pages:", len(_pdf.pages))
            for _pn, _pg in enumerate(_pdf.pages[:6]):
                _tbls = _pg.extract_tables()
                st.write("PAGE", _pn + 1, "— tables:", len(_tbls))
                for _ti, _tbl in enumerate(_tbls):
                    _nc = len(_tbl[0]) if _tbl else 0
                    st.write("  table", _ti, "rows:", len(_tbl), "cols:", _nc)
                    for _ri in range(min(4, len(_tbl))):
                        _cells = [str(_tbl[_ri][_ci] or "").strip()[:25] for _ci in range(_nc)]
                        st.write("   ", _ri, _cells)
        st.stop()

        df = extract_df_from_pdf(uploaded, pdf_password)
        st.info("DEBUG raw PDF cols: " + str(list(df.columns)) + " | rows: " + str(len(df)))  # TEMP DEBUG
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
