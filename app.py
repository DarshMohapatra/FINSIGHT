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
if "forecast_cache" not in st.session_state:
    st.session_state.forecast_cache = None
if "forecast_df_id" not in st.session_state:
    st.session_state.forecast_df_id = None


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

# ── SmartCash globals ──────────────────────────────────────
import json as _scj, urllib.request as _scu

@st.cache_data(ttl=3600)
def load_card_databases():
    _b = "https://raw.githubusercontent.com/DarshMohapatra/FINSIGHT/main/"
    def _g(f):
        try:
            with _scu.urlopen(_b+f, timeout=5) as resp: return _scj.load(resp)
        except Exception: return []
    return _g("card_master.json"), _g("card_rewards.json")

SC_CARD_MASTER, SC_CARD_REWARDS = load_card_databases()
SC_RWD  = {r["card_id"]: r["rates"] for r in SC_CARD_REWARDS}
SC_NAME = {c["card_id"]: c["bank"]+" "+c["card_name"] for c in SC_CARD_MASTER}
SC_CMAP = {
    "Food & Dining":"Food & Dining", "Grocery":"Grocery",
    "Shopping":"Shopping", "Travel":"Travel", "Fuel":"Fuel",
    "Healthcare":"Healthcare", "Entertainment":"Entertainment",
    "Utility":"Utility", "Salary":"Other", "Other":"Other"}

# ── LifeEvent Radar globals ──────────────────────────────────
import re as _re, datetime as _dt

@st.cache_data(ttl=86400)
def load_le_databases():
    _b = "https://raw.githubusercontent.com/DarshMohapatra/FINSIGHT/main/"
    def _gj(f):
        try:
            with _scu.urlopen(_b+f, timeout=8) as resp: return _scj.load(resp)
        except Exception: return []
    def _gc(f):
        try:
            import io, csv
            with _scu.urlopen(_b+f, timeout=8) as resp:
                content = resp.read().decode("utf-8")
            reader = csv.DictReader(io.StringIO(content))
            return list(reader)
        except Exception: return []
    sigs      = _gj("life_event_signatures.json")
    checklists= _gj("event_checklist_templates.json")
    merchants = _gc("merchant_event_map.csv")
    return sigs, checklists, merchants

LE_SIGS, LE_CHECKLISTS, LE_MERCHANTS = load_le_databases()
LE_KW_LOOKUP = {}
for _row in LE_MERCHANTS:
    _kw = str(_row.get("keyword","")).lower().strip()
    if _kw:
        if _kw not in LE_KW_LOOKUP: LE_KW_LOOKUP[_kw] = []
        LE_KW_LOOKUP[_kw].append((_row["event_id"], int(_row.get("signal_strength",3))))
LE_SIG_LOOKUP  = {s["event_id"]: s for s in LE_SIGS}
LE_CL_LOOKUP   = {c["event_id"]: c for c in LE_CHECKLISTS}

def le_run_detection(df, user_profile=None):
    from datetime import timedelta
    analysis_date = df["DATE"].max()
    window_start  = analysis_date - timedelta(days=60)
    recent_start  = analysis_date - timedelta(days=30)
    df_win = df[(df["DATE"]>=window_start)&(df["DATE"]<=analysis_date)].copy()
    def get_tags(desc):
        if pd.isna(desc): return []
        d = str(desc).lower()
        tags = set()
        for kw,matches in LE_KW_LOOKUP.items():
            if kw in d:
                for eid,strength in matches: tags.add((eid,strength))
        return list(tags)
    df_win["_tags"] = df_win["TRANSACTION DETAILS"].apply(get_tags)
    results = []
    for sig in LE_SIGS:
        eid = sig["event_id"]
        # S1 velocity
        def esp(pdf):
            t=0
            for _,row in pdf.iterrows():
                for teid,st in row["_tags"]:
                    if teid==eid: t+=row["WITHDRAWAL AMT"]*(st/5)
            return t
        rec = esp(df_win[df_win["DATE"]>=recent_start])
        pri = esp(df_win[df_win["DATE"]<recent_start])
        thr = sig["spend_acceleration_threshold"]
        if pri==0 and rec>0: s1=min(85,50+rec/1000)
        elif pri==0: s1=0
        else:
            acc=rec/(pri+1)
            if acc>=thr*2: s1=100
            elif acc>=thr: s1=60+(acc/thr)*20
            elif acc>=thr*0.5: s1=30+(acc/thr)*30
            else: s1=min(25,acc*10)
        s1=round(min(100,max(0,s1)),1)
        # S2 NLP
        cutoff = analysis_date - timedelta(days=180)
        df_scan= df[df["DATE"]>=cutoff]
        raw_s2=0.0
        hits=[]
        for _,row in df_scan.iterrows():
            desc=str(row.get("TRANSACTION DETAILS","")).lower()
            days_ago=(analysis_date-row["DATE"]).days
            rw=max(0.3,1.0-(days_ago/180)*0.7)
            for kw,matches in LE_KW_LOOKUP.items():
                if kw in desc:
                    for teid,strength in matches:
                        if teid==eid:
                            raw_s2+=strength*rw*4
                            hits.append({"keyword":kw,"desc":str(row["TRANSACTION DETAILS"])[:60],"amount":row["WITHDRAWAL AMT"]})
        s2=round(min(100,(raw_s2/100)*100),1)
        # S3 profile
        if user_profile:
            try: age=int(str(user_profile.get("age_range","25-34")).split("-")[0])+5
            except: age=30
            ms=user_profile.get("marital_status","unknown").lower()
            ch=user_profile.get("children_count",0)
            ho=user_profile.get("home_owner",False)
            s3_map={"MARRIAGE":50,"BABY":50,"HOME_PURCHASE":50,"JOB_SWITCH":50,
                    "CHILD_SCHOOL":50,"RELOCATION":50,"MEDICAL_EVENT":50,"RETIREMENT_PREP":50}
            if eid=="MARRIAGE":
                s3_map[eid]=75 if ms in ["single","unmarried"] and 22<=age<=35 else (15 if ms=="married" else 50)
            elif eid=="BABY": s3_map[eid]=65 if ms=="married" and 25<=age<=38 else 30
            elif eid=="HOME_PURCHASE": s3_map[eid]=65 if not ho and 28<=age<=45 else 25
            elif eid=="JOB_SWITCH": s3_map[eid]=60 if 24<=age<=40 else 30
            elif eid=="CHILD_SCHOOL": s3_map[eid]=65 if ch>0 else 20
            elif eid=="MEDICAL_EVENT": s3_map[eid]=65 if age>=50 else (55 if age>=40 else 50)
            elif eid=="RETIREMENT_PREP": s3_map[eid]=70 if age>=45 else (55 if age>=38 else 25)
            s3=s3_map[eid]
        else: s3=50
        # S4 seasonal
        recent90=df[df["DATE"]>=analysis_date-timedelta(days=90)]
        active_months=set(recent90["DATE"].dt.month.tolist())
        seasonal=set(sig["seasonal_months"])
        overlap=len(active_months&seasonal)
        cur_bonus=20 if analysis_date.month in seasonal else 0
        s4=round(min(100,(overlap/max(len(seasonal),1))*80+cur_bonus),1)
        # Composite
        composite=round(s1*0.40+s2*0.25+s3*0.25+s4*0.10,1)
        alert="🔴 HARD" if composite>=82 else ("🟡 SOFT" if composite>=65 else "")
        results.append({"event_id":eid,"event_name":sig["event_name"],
                         "emoji":sig["emoji"],"composite":composite,
                         "s1":s1,"s2":s2,"s3":s3,"s4":s4,
                         "alert":alert,"hits":hits[:3]})
    results.sort(key=lambda x:x["composite"],reverse=True)
    return results


def sc_best(amount, category, wallet):
    cat = SC_CMAP.get(category, "Other")
    bi, br, bc = "NONE", 0.0, 0.0
    for cid in wallet:
        rate = SC_RWD.get(cid, {}).get(cat, SC_RWD.get(cid, {}).get("Other", 0))
        cash = round(amount * rate / 100, 2)
        if cash > bc: bi, br, bc = cid, rate, cash
    return {"name": SC_NAME.get(bi, bi), "rate": br,
            "cash": bc, "base": round(amount/100, 2)}

def sc_analyse(df, wallet):
    sp = df[(df["WITHDRAWAL AMT"]>0) & df["CATEGORY"].notna()
            & (df["CATEGORY"]!="Salary")].copy()
    rows = []
    for _, row in sp.iterrows():
        res = sc_best(row["WITHDRAWAL AMT"], row["CATEGORY"], wallet)
        rows.append({"DATE": row["DATE"],
                     "DESCRIPTION": row["TRANSACTION DETAILS"],
                     "CATEGORY": row["CATEGORY"],
                     "AMOUNT": row["WITHDRAWAL AMT"],
                     "BEST_CARD": res["name"],
                     "BEST_RATE": res["rate"],
                     "BEST_CASHBACK": res["cash"],
                     "BASELINE": res["base"],
                     "EXTRA": round(res["cash"]-res["base"], 2)})
    return pd.DataFrame(rows)

def sc_summary(rdf):
    return (rdf.groupby("CATEGORY")
            .agg(spend=("AMOUNT","sum"), cashback=("BEST_CASHBACK","sum"),
                 txns=("AMOUNT","count"),
                 best_card=("BEST_CARD", lambda x: x.mode()[0]),
                 avg_rate=("BEST_RATE","mean"))
            .assign(extra=lambda x: x["cashback"]-x["cashback"]/x["avg_rate"]
                    if False else x["cashback"]-x["spend"]/100)
            .sort_values("spend", ascending=False).reset_index())



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
    """
    Universal bank statement PDF parser.
    Supports:
      - 2-col format: DATE NARRATION AMOUNT BALANCE         (ICICI, HDFC, Axis, Kotak)
      - 3-col format: DATE NARRATION DEBIT CREDIT BALANCE   (SBI, PNB, Canara)
    Date formats: DD-MM-YYYY, DD/MM/YYYY, YYYY-MM-DD, DD-Mon-YYYY, DD Mon YYYY, DD/MM/YY
    Deposit vs withdrawal: balance delta (100% accurate), keyword fallback if ambiguous.
    """
    import pdfplumber
    import pandas as pd
    import re as _re
    from datetime import datetime

    DATE_FORMATS = [
        (_re.compile(r"^\d{2}-\d{2}-\d{4}"),        "%d-%m-%Y"),
        (_re.compile(r"^\d{2}/\d{2}/\d{4}"),        "%d/%m/%Y"),
        (_re.compile(r"^\d{4}-\d{2}-\d{2}"),        "%Y-%m-%d"),
        (_re.compile(r"^\d{4}/\d{2}/\d{2}"),        "%Y/%m/%d"),
        (_re.compile(r"^\d{2}-[A-Za-z]{3}-\d{4}"),  "%d-%b-%Y"),
        (_re.compile(r"^\d{2}\s[A-Za-z]{3}\s\d{4}"),"%d %b %Y"),
        (_re.compile(r"^\d{2}/\d{2}/\d{2}"),        "%d/%m/%y"),
        (_re.compile(r"^\d{2}-\d{2}-\d{2}"),        "%d-%m-%y"),
    ]
    AMT_PAT = _re.compile(r"[\d,]+\.\d{2}")

    def parse_date(s):
        for pat, fmt in DATE_FORMATS:
            m = pat.match(s)
            if m:
                token = m.group(0)
                try:
                    datetime.strptime(token, fmt)
                    return token, fmt
                except ValueError:
                    continue
        return None

    def to_float(s):
        try:
            return float(str(s).replace(",", "").strip())
        except Exception:
            return None

    open_kwargs = {"password": password} if password else {}
    try:
        pdf_file = pdfplumber.open(pdf_path, **open_kwargs)
        pdf_file.close()
    except Exception as e:
        err = str(e).lower()
        if any(k in err for k in ["password", "decrypt", "encrypt", "incorrect"]):
            raise ValueError(
                "❌ Incorrect PDF password. "
                "Try your PAN number, date of birth (DDMMYYYY), or account number."
            )
        raise

    # ── Detect 3-column format from table header ──────────────────────────────
    three_col_mode = False
    with pdfplumber.open(pdf_path, **open_kwargs) as pdf:
        for page in pdf.pages[:3]:
            for tbl in (page.extract_tables() or []):
                if tbl:
                    header = [str(c or "").strip().upper() for c in tbl[0]]
                    has_debit  = any("DEBIT" in h or h == "DR" or "WITHDRAWAL" in h for h in header)
                    has_credit = any("CREDIT" in h or h == "CR" or "DEPOSIT" in h for h in header)
                    if has_debit and has_credit:
                        three_col_mode = True
                        break
            if three_col_mode:
                break

    # ── Parse text lines ──────────────────────────────────────────────────────
    raw_rows = []
    with pdfplumber.open(pdf_path, **open_kwargs) as pdf:
        for page in pdf.pages:
            try:
                text = page.extract_text(x_tolerance=2, y_tolerance=2)
            except Exception:
                continue
            if not text:
                continue
            for raw in text.splitlines():
                line = raw.strip()
                if not line:
                    continue
                result = parse_date(line)
                if not result:
                    continue
                date_token, date_fmt = result
                rest    = line[len(date_token):].strip()
                amounts = AMT_PAT.findall(rest)
                if len(amounts) < 2:
                    continue   # 0 = no amounts, 1 = B/F opening balance — skip both
                first_pos = rest.index(amounts[0])
                narration = rest[:first_pos].strip()
                raw_rows.append((date_token, date_fmt, narration, amounts))

    if not raw_rows:
        raise ValueError(
            "No transactions found in this PDF. "
            "It may be image-only (scanned). Try a CSV/Excel export."
        )

    # ── Assign deposit / withdrawal ───────────────────────────────────────────
    CREDIT_KWS = ["REFUND", "SALARY", "INTEREST", "CASHBACK", "REVERSAL",
                  "REWARD", "DIVIDEND", "/CR", " CR ", "NEFT CR", "IMPS CR", "RTGS CR"]

    rows     = []
    prev_bal = None

    for date_token, date_fmt, narration, amounts in raw_rows:
        deposit    = 0.0
        withdrawal = 0.0
        balance    = to_float(amounts[-1]) or 0.0

        # 3-col mode AND enough amounts present
        if three_col_mode and len(amounts) >= 3:
            debit_cand  = to_float(amounts[-3])
            credit_cand = to_float(amounts[-2])
            bal_cand    = balance
            tol = 2.0
            if prev_bal is not None:
                if debit_cand and abs(round(prev_bal - debit_cand, 2) - round(bal_cand, 2)) <= tol:
                    withdrawal = debit_cand
                elif credit_cand and abs(round(prev_bal + credit_cand, 2) - round(bal_cand, 2)) <= tol:
                    deposit = credit_cand
                else:
                    withdrawal = debit_cand  or 0.0
                    deposit    = credit_cand or 0.0
            else:
                withdrawal = debit_cand  or 0.0
                deposit    = credit_cand or 0.0

        else:
            # 2-col mode (or 3-col with only 2 amounts on this line)
            amt_val = to_float(amounts[-2])
            tol = 2.0
            if prev_bal is not None and amt_val is not None:
                exp_wd  = round(prev_bal - amt_val, 2)
                exp_dep = round(prev_bal + amt_val, 2)
                bal_r   = round(balance, 2)
                if abs(bal_r - exp_wd) <= tol:
                    withdrawal = amt_val
                elif abs(bal_r - exp_dep) <= tol:
                    deposit = amt_val
                else:
                    # Ambiguous — keyword fallback
                    if any(k in narration.upper() for k in CREDIT_KWS):
                        deposit = amt_val
                    else:
                        withdrawal = amt_val
            else:
                if any(k in narration.upper() for k in CREDIT_KWS):
                    deposit = to_float(amounts[-2]) or 0.0
                else:
                    withdrawal = to_float(amounts[-2]) or 0.0

        prev_bal = balance
        rows.append({
            "DATE":                date_token,
            "TRANSACTION DETAILS": narration,
            "DEPOSIT AMT":         round(deposit,    2),
            "WITHDRAWAL AMT":      round(withdrawal, 2),
            "BALANCE AMT":         round(balance,    2),
        })

    df = pd.DataFrame(rows)

    # Parse dates — try each format, use whichever matches >80% of rows
    for _, fmt in DATE_FORMATS:
        converted = pd.to_datetime(df["DATE"], format=fmt, errors="coerce")
        if converted.notna().sum() > len(df) * 0.8:
            df["DATE"] = converted
            break
    else:
        df["DATE"] = pd.to_datetime(df["DATE"], errors="coerce")

    df = df.dropna(subset=["DATE"])
    df = df.sort_values("DATE").reset_index(drop=True)
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
                          .str.replace("Rs.", "", regex=False).str.replace("₹", "", regex=False)
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
    if raw >= 1e7:   return "₹" + str(round(x/1e7, 1)) + "Cr"
    elif raw >= 1e5: return "₹" + str(round(x/1e5, 1)) + "L"
    elif raw >= 1e3: return "₹" + str(round(x/1e3, 1)) + "K"
    else:            return "₹" + str(round(x, 0))[:-2]


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

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["📤  UPLOAD", "📊  DASHBOARD", "📈  FORECAST", "🤖  AI ADVISOR", "💳  SMARTCASH", "🎯  LIFEEVENT"])

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
                            try:
                                df = process_file(uploaded, pdf_password)
                            except ValueError as _ve:
                                _msg = str(_ve)
                                if 'password' in _msg.lower() or '❌' in _msg:
                                    st.error(_msg)
                                    st.stop()
                                else:
                                    raise
                            st.session_state.user_df = df
                            st.session_state.analysis_done = True
                            # Invalidate forecast cache so it retrains on new data
                            st.session_state.forecast_cache = None
                            st.session_state.forecast_df_id = None
                            # Invalidate SmartCash results from previous file
                            st.session_state.pop("sc_results", None)
                            st.session_state.pop("sc_cat", None)
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
            # Use a fingerprint of the current data to detect when user uploads new file
            _df_id = (len(df), df["WITHDRAWAL AMT"].sum(), df["DATE"].min(), df["DATE"].max())
            _need_retrain = (st.session_state.forecast_cache is None
                            or st.session_state.forecast_df_id != _df_id)
            if _need_retrain:
                with st.spinner("Forecasting from your spending data..."):
                    try:
                        # Build monthly spending from user's actual data
                        df_fc = df[df["IS_ANOMALY"] == 0] if "IS_ANOMALY" in df.columns else df
                        ms = df_fc.groupby(df_fc["DATE"].dt.to_period("M"))["WITHDRAWAL AMT"].sum().reset_index()
                        ms.columns = ["ds", "y"]
                        ms["ds"] = ms["ds"].dt.to_timestamp()
                        ms = ms.sort_values("ds").reset_index(drop=True)

                        n = len(ms)
                        values = ms["y"].values
                        last_date = ms["ds"].max()

                        # Linear trend line through all months
                        x = np.arange(n, dtype=float)
                        if n >= 2:
                            slope, intercept = np.polyfit(x, values, 1)
                        else:
                            slope, intercept = 0.0, values[0]

                        # Seasonal ratio per calendar month (Jan=1..Dec=12)
                        # How much each calendar month deviates from overall avg
                        overall_avg = values.mean()
                        ms["cal_month"] = ms["ds"].dt.month
                        cal_avg = ms.groupby("cal_month")["y"].mean()
                        seasonal = {}
                        for m_num in range(1, 13):
                            if m_num in cal_avg.index and overall_avg > 0:
                                seasonal[m_num] = cal_avg[m_num] / overall_avg
                            else:
                                seasonal[m_num] = 1.0

                        # Project next 6 months
                        future_dates = pd.date_range(last_date + pd.DateOffset(months=1), periods=6, freq="MS")
                        predicted = []
                        for i, fdate in enumerate(future_dates):
                            # Trend: extrapolate the trend line
                            trend_val = intercept + slope * (n + i)
                            # Seasonal: adjust by calendar month ratio
                            s_ratio = seasonal[fdate.month]
                            pred = max(trend_val * s_ratio, 0)
                            predicted.append(pred)

                        # Confidence band: +/- std dev of residuals, widening over time
                        if n >= 2:
                            trend_fitted = intercept + slope * x
                            residuals = values - trend_fitted
                            std_y = residuals.std()
                        else:
                            std_y = overall_avg * 0.1
                        lower = [max(p - std_y * (0.8 + 0.2 * i), 0) for i, p in enumerate(predicted)]
                        upper = [p + std_y * (0.8 + 0.2 * i) for i, p in enumerate(predicted)]

                        fc = pd.DataFrame({
                            "ds": future_dates,
                            "yhat": predicted,
                            "yhat_lower": lower,
                            "yhat_upper": upper,
                        })
                        st.session_state.forecast_cache = {"ms": ms, "fc": fc}
                        st.session_state.forecast_df_id = _df_id
                    except Exception as e:
                        st.error("Forecast error: " + str(e))
                        st.session_state.forecast_cache = None
            if st.session_state.forecast_cache is not None:
                ms = st.session_state.forecast_cache["ms"]
                fc = st.session_state.forecast_cache["fc"]
                try:
                    fig2, ax2 = dark_fig(2, 1, (14, 10))
                    ax2[0].plot(ms["ds"], ms["y"], color="#00c8e0", linewidth=2.5, label="Actual")
                    # Combine actual + forecast for a continuous line
                    all_ds = pd.concat([ms[["ds"]], fc[["ds"]]]).reset_index(drop=True)
                    all_yhat = list(ms["y"].values) + list(fc["yhat"].values)
                    ax2[0].plot(all_ds["ds"], all_yhat, color="#00f0a0", linewidth=2, linestyle="--", label="Forecast")
                    ax2[0].fill_between(fc["ds"], fc["yhat_lower"], fc["yhat_upper"], alpha=0.12, color="#00f0a0")
                    ax2[0].axvline(x=ms["ds"].max(), color="#ff3c64", linestyle="--", linewidth=1.5, alpha=0.7, label="Forecast Start")
                    ax2[0].set_title("Monthly Withdrawal Forecast", fontsize=13, pad=14)
                    ax2[0].yaxis.set_major_formatter(mticker.FuncFormatter(cfmt))
                    ax2[0].legend(facecolor="#080d1a", labelcolor="#c8d0e0", fontsize=9)
                    fo = fc
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


    with tab5:
        st.markdown('<div style="padding:32px 40px 0"><div style="font-family:DM Mono,monospace;font-size:10px;color:#00f5a0;letter-spacing:3px;margin-bottom:8px">FEATURE 01 — SMARTCASH</div><div style="font-size:26px;font-weight:800;margin-bottom:8px">💳 Card Reward Maximiser</div><div style="color:rgba(255,255,255,0.4);font-size:14px;margin-bottom:24px">Find the best card in your wallet for every rupee you spend.</div></div>', unsafe_allow_html=True)
        if st.session_state.get("user_df") is None:
            st.info("Upload your bank statement in the UPLOAD tab first.")
        elif not SC_CARD_MASTER:
            st.error("Card database failed to load — check card_master.json on GitHub.")
        else:
            df_sc = st.session_state["user_df"]
            card_opts = {c["bank"]+" — "+c["card_name"]+" ("+("FREE" if c["annual_fee"]==0 else "₹"+str(c["annual_fee"])+"/yr")+")": c["card_id"] for c in SC_CARD_MASTER}
            import re as _re
            _wdr = df_sc[df_sc["WITHDRAWAL AMT"]>0]["TRANSACTION DETAILS"].astype(str).str.upper()
            _all_txn_text = " ".join(_wdr.tolist())
            _auto = []
            _all_keys = list(card_opts.keys())
            # Broad patterns: match bank names in any context (UPI handles, NEFT, bill pay, etc.)
            _bank_patterns = [
                # HDFC — matches HDFCBK, HDFC BANK, HDFC CC, @hdfc, HDFC CREDIT, etc.
                (r"HDFC|HDFCBK|@HDFC",              "HDFC Bank"),
                # Axis — matches AXIS BANK, AXISB, @axl, AXIS CC, etc.
                (r"AXIS\s*BANK|AXISB|@AXIS|@AXL|AXIS.*(CC|CARD|CREDIT|CRD)", "Axis Bank"),
                # ICICI — matches ICICI, ICICIB, @icici, ICICI CC, etc.
                (r"ICICI|ICICIB|@ICICI",             "ICICI Bank"),
                # SBI — matches SBICARD, SBI CARD, SBICRD, SBI CC, @SBI
                (r"SBI\s*CARD|SBICRD|SBI.*(CC|CREDIT|CRD)|@SBI", "SBI Card"),
                # Kotak — matches KOTAK, @kotak, KOTAK CC, etc.
                (r"KOTAK|@KOTAK",                    "Kotak Mahindra Bank"),
                # IDFC — matches IDFC, IDFCFIRST, @IDFCFIRST, etc.
                (r"IDFC|IDFCFIRST|@IDFC",           "IDFC FIRST Bank"),
                # Yes Bank
                (r"YES\s*BANK|YESBK|@YBL|@YESBANK", "Yes Bank"),
                # RBL Bank
                (r"RBL\s*BANK|RBLBANK|@RBL",        "RBL Bank"),
                # Amex
                (r"AMEX|AMERICAN\s*EXPRESS|AMERICANEXPRESS", "American Express"),
                # Generic credit card payment keywords — try to extract bank from context
                (r"CRED\b.*HDFC|HDFC.*CRED\b",      "HDFC Bank"),
                (r"CRED\b.*AXIS|AXIS.*CRED\b",      "Axis Bank"),
                (r"CRED\b.*ICICI|ICICI.*CRED\b",    "ICICI Bank"),
                (r"CRED\b.*SBI|SBI.*CRED\b",        "SBI Card"),
            ]
            for _pat, _bank in _bank_patterns:
                if _re.search(_pat, _all_txn_text):
                    for _k in _all_keys:
                        if _k.startswith(_bank) and _k not in _auto:
                            _auto.append(_k)
            # Also check UPI VPA handles (e.g., @hdfcbank, @icici, @sbi, @axisbank)
            _vpa_bank_map = {
                "hdfcbank": "HDFC Bank", "hdfc": "HDFC Bank",
                "icici": "ICICI Bank", "icicibank": "ICICI Bank",
                "axisbank": "Axis Bank", "axis": "Axis Bank", "axl": "Axis Bank",
                "sbi": "SBI Card", "sbicard": "SBI Card",
                "kotak": "Kotak Mahindra Bank", "kotakbank": "Kotak Mahindra Bank",
                "idfcfirst": "IDFC FIRST Bank", "idfc": "IDFC FIRST Bank",
                "yesbank": "Yes Bank", "ybl": "Yes Bank",
                "rbl": "RBL Bank", "rblbank": "RBL Bank",
            }
            _vpa_matches = _re.findall(r"@([A-Z0-9]+)", _all_txn_text)
            for _vpa in _vpa_matches:
                _vpa_lower = _vpa.lower()
                _bank = _vpa_bank_map.get(_vpa_lower)
                if _bank:
                    for _k in _all_keys:
                        if _k.startswith(_bank) and _k not in _auto:
                            _auto.append(_k)
            if _auto:
                st.markdown(f'<div style="margin-bottom:10px;padding:10px 14px;background:rgba(0,245,160,0.06);border:1px solid rgba(0,245,160,0.2);border-radius:8px;font-size:12px;color:#00f5a0">⚡ Auto-detected {len(_auto)} card(s) from your statement — confirm or edit below</div>', unsafe_allow_html=True)
            sel = st.multiselect("Select credit cards you own:", _all_keys, default=_auto, key="sc_wallet_sel")
            wallet_sc = [card_opts[s] for s in sel]
            run_sc = st.button("⚡ Analyse Cashback Potential", key="sc_run_btn", disabled=(len(wallet_sc)==0))
            if run_sc and wallet_sc:
                with st.spinner("Calculating best card for every transaction..."):
                    st.session_state["sc_results"] = sc_analyse(df_sc, wallet_sc)
                    st.session_state["sc_cat"]     = sc_summary(st.session_state["sc_results"])
            if st.session_state.get("sc_results") is not None:
                rdf  = st.session_state["sc_results"]
                scat = st.session_state["sc_cat"]
                tot_sp = rdf["AMOUNT"].sum()
                tot_cb = rdf["BEST_CASHBACK"].sum()
                tot_ex = rdf["EXTRA"].sum()
                eff_r  = (tot_cb/tot_sp*100) if tot_sp>0 else 0
                c1,c2,c3,c4 = st.columns(4)
                c1.markdown(f'<div class="metric-card"><div class="metric-lbl">TOTAL SPEND</div><div class="metric-val">₹{tot_sp:,.0f}</div></div>', unsafe_allow_html=True)
                c2.markdown(f'<div class="metric-card"><div class="metric-lbl">BEST CASHBACK</div><div class="metric-val">₹{tot_cb:,.0f}</div></div>', unsafe_allow_html=True)
                c3.markdown(f'<div class="metric-card"><div class="metric-lbl">EXTRA vs 1% CARD</div><div class="metric-val">₹{tot_ex:,.0f}</div></div>', unsafe_allow_html=True)
                c4.markdown(f'<div class="metric-card"><div class="metric-lbl">EFFECTIVE RATE</div><div class="metric-val">{eff_r:.2f}%</div></div>', unsafe_allow_html=True)
                st.markdown("<div style='margin:20px 0 8px'><b>Category Guide — Best card per spend type:</b></div>", unsafe_allow_html=True)
                for _, crow in scat.iterrows():
                    pct = min(int(crow["avg_rate"]/6*100), 100)
                    st.markdown(f'<div style="margin-bottom:8px;padding:14px;background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.07);border-radius:10px"><div style="display:flex;justify-content:space-between;margin-bottom:6px"><b>{crow["CATEGORY"]}</b><span style="color:#00f5a0;font-size:11px;font-family:DM Mono,monospace">Use → {crow["best_card"]}</span></div><div style="color:rgba(255,255,255,0.4);font-size:12px;margin-bottom:8px">₹{crow["spend"]:,.0f} spent · {crow["txns"]:,} txns · earns ₹{crow["cashback"]:,.0f} at {crow["avg_rate"]:.1f}%</div><div style="background:rgba(255,255,255,0.06);border-radius:3px;height:3px"><div style="background:linear-gradient(90deg,#00f5a0,#00d4ff);width:{pct}%;height:3px;border-radius:3px"></div></div></div>', unsafe_allow_html=True)
                st.markdown("<div style='margin:20px 0 8px'><b>Top 20 Transactions — Best card to use:</b></div>", unsafe_allow_html=True)
                top20 = rdf.nlargest(20,"BEST_CASHBACK")[["DATE","DESCRIPTION","CATEGORY","AMOUNT","BEST_CARD","BEST_RATE","BEST_CASHBACK"]].copy()
                top20["AMOUNT"]        = top20["AMOUNT"].apply(lambda x: f"₹{x:,.0f}")
                top20["BEST_CASHBACK"] = top20["BEST_CASHBACK"].apply(lambda x: f"₹{x:,.0f}")
                top20["BEST_RATE"]     = top20["BEST_RATE"].apply(lambda x: f"{x:.1f}%")
                st.dataframe(top20, use_container_width=True, hide_index=True)


    with tab6:
        st.markdown('<div style="padding:32px 40px 0"><div style="font-family:DM Mono,monospace;font-size:10px;color:#00f5a0;letter-spacing:3px;margin-bottom:8px">FEATURE 07 — LIFEEVENT RADAR</div><div style="font-size:26px;font-weight:800;margin-bottom:8px">🎯 Life Event Radar</div><div style="color:rgba(255,255,255,0.4);font-size:14px;margin-bottom:24px">Detects life events from your spending patterns and generates personalised financial checklists.</div></div>', unsafe_allow_html=True)
        if st.session_state.get("user_df") is None:
            st.info("Upload your bank statement in the UPLOAD tab first.")
        elif not LE_SIGS:
            st.error("LifeEvent database not loaded — check GitHub for life_event_signatures.json")
        else:
            df_le = st.session_state["user_df"]
            # ── User Profile ────────────────────────────────────────────
            with st.expander("📋 Your Profile (improves detection accuracy)", expanded=False):
                st.markdown('<div style="font-size:12px;color:rgba(255,255,255,0.4);margin-bottom:12px">Takes 30 seconds — boosts detection accuracy by 40%</div>', unsafe_allow_html=True)
                pc1,pc2,pc3 = st.columns(3)
                age_r  = pc1.selectbox("Age Range",["18-24","25-34","35-44","45-54","55+"],index=1,key="le_age")
                marital= pc2.selectbox("Marital Status",["Single","Married","Divorced"],index=0,key="le_marital")
                kids   = pc3.number_input("No. of Children",0,10,0,key="le_kids")
                pc4,pc5,pc6 = st.columns(3)
                home   = pc4.selectbox("Home Ownership",["Renting","Own Home"],index=0,key="le_home")
                emp    = pc5.selectbox("Employment",["Salaried","Self-Employed","Business"],index=0,key="le_emp")
                city   = pc6.selectbox("City Tier",["Tier 1","Tier 2","Tier 3"],index=0,key="le_city")
                user_profile_le = {"age_range":age_r,"marital_status":marital.lower(),
                    "children_count":kids,"home_owner":(home=="Own Home"),
                    "employment_type":emp.lower(),"city_tier":int(city.split()[1])}
            user_profile_le = {"age_range":st.session_state.get("le_age","25-34"),
                "marital_status":st.session_state.get("le_marital","single").lower(),
                "children_count":st.session_state.get("le_kids",0),
                "home_owner":(st.session_state.get("le_home","Renting")=="Own Home"),
                "employment_type":st.session_state.get("le_emp","salaried").lower(),
                "city_tier":1}
            # ── Run Detection ────────────────────────────────────────────
            if st.button("🎯 Scan for Life Events", key="le_scan"):
                with st.spinner("Scanning your spending patterns for life event signals..."):
                    st.session_state["le_results"] = le_run_detection(df_le, user_profile_le)
            if st.session_state.get("le_results"):
                le_res = st.session_state["le_results"]
                alerts = [r for r in le_res if r["composite"]>=65]
                if not alerts:
                    st.markdown('<div style="margin:24px 40px;padding:24px;background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.07);border-radius:12px;text-align:center;color:rgba(255,255,255,0.4)">✅ No significant life events detected in your spending — all patterns look routine.</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div style="margin:16px 40px 8px;font-family:DM Mono,monospace;font-size:10px;color:#00f5a0;letter-spacing:2px">{len(alerts)} LIFE EVENT(S) DETECTED</div>', unsafe_allow_html=True)
                    for ev in alerts:
                        cl = LE_CL_LOOKUP.get(ev["event_id"],{})
                        alert_col = "#ff3c64" if ev["alert"]=="🔴 HARD" else "#f5a000"
                        st.markdown(f'<div style="margin:0 40px 16px;padding:20px 24px;background:rgba(255,255,255,0.03);border:1px solid {alert_col}40;border-radius:14px"><div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:12px"><div style="font-size:20px;font-weight:800">{ev["emoji"]} {ev["event_name"]}</div><div style="font-family:DM Mono,monospace;font-size:12px;color:{alert_col};background:{alert_col}20;padding:4px 10px;border-radius:6px">{ev["alert"]} · {ev["composite"]:.0f}/100</div></div><div style="display:grid;grid-template-columns:1fr 1fr 1fr 1fr;gap:8px;margin-bottom:12px"><div style="text-align:center;padding:8px;background:rgba(255,255,255,0.04);border-radius:8px"><div style="font-size:10px;color:rgba(255,255,255,0.4);font-family:DM Mono,monospace">VELOCITY</div><div style="font-weight:700;color:#00f5a0">{ev["s1"]:.0f}</div></div><div style="text-align:center;padding:8px;background:rgba(255,255,255,0.04);border-radius:8px"><div style="font-size:10px;color:rgba(255,255,255,0.4);font-family:DM Mono,monospace">NLP</div><div style="font-weight:700;color:#00d4ff">{ev["s2"]:.0f}</div></div><div style="text-align:center;padding:8px;background:rgba(255,255,255,0.04);border-radius:8px"><div style="font-size:10px;color:rgba(255,255,255,0.4);font-family:DM Mono,monospace">PROFILE</div><div style="font-weight:700;color:#a78bfa">{ev["s3"]:.0f}</div></div><div style="text-align:center;padding:8px;background:rgba(255,255,255,0.04);border-radius:8px"><div style="font-size:10px;color:rgba(255,255,255,0.4);font-family:DM Mono,monospace">SEASONAL</div><div style="font-weight:700;color:#f59e0b">{ev["s4"]:.0f}</div></div></div>', unsafe_allow_html=True)
                        if ev["hits"]:
                            st.markdown(f'<div style="margin:0 40px 4px;padding:0 24px;font-size:12px;color:rgba(255,255,255,0.5)">Top signals: {", ".join([h["keyword"] for h in ev["hits"]])}</div>', unsafe_allow_html=True)
                        if cl:
                            with st.expander(f"📋 Financial Checklist — {ev['event_name']}"):
                                st.markdown(f'**Corpus needed:** {cl.get("corpus_estimate","—")}')
                                st.markdown(f'**Timeline:** {cl.get("timeline_months","?")} months')
                                p1 = [x for x in cl.get("checklist_items",[]) if x["priority"]==1]
                                p2 = [x for x in cl.get("checklist_items",[]) if x["priority"]==2]
                                if p1:
                                    st.markdown("**🔴 Do immediately:**")
                                    for item in p1: st.checkbox(f'[{item["category"]}] {item["action"]}', key=f'le_{ev["event_id"]}_{item["priority"]}_{item["action"][:20]}')
                                if p2:
                                    st.markdown("**🟡 Do within 3 months:**")
                                    for item in p2: st.checkbox(f'[{item["category"]}] {item["action"]}', key=f'le2_{ev["event_id"]}_{item["action"][:20]}')
                                if cl.get("insurance_gaps"):
                                    st.markdown("**🛡️ Insurance gaps to fill:**")
                                    for gap in cl["insurance_gaps"]: st.markdown(f"- {gap}")
                # ── Full scores table ────────────────────────────────────────
                with st.expander("🔍 Full Detection Scores (all 8 events)"):
                    score_df = pd.DataFrame([{"Event":r["emoji"]+" "+r["event_name"],"Score":r["composite"],"Velocity":r["s1"],"NLP":r["s2"],"Profile":r["s3"],"Seasonal":r["s4"],"Alert":r["alert"] or "—"} for r in le_res])
                    st.dataframe(score_df, use_container_width=True, hide_index=True)

    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown('<div style="border-top:1px solid rgba(255,255,255,0.06);margin-top:60px;padding:24px 40px;"><span style="font-family:DM Mono,monospace;font-size:10px;color:rgba(255,255,255,0.15);letter-spacing:2px;">FINSIGHT · ISOLATION FOREST + TREND FORECAST + LLAMA 3.3 70B</span></div>', unsafe_allow_html=True)