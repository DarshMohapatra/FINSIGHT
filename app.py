import streamlit as st
import re
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import os
from groq import Groq

# ── Supabase auth ────────────────────────────────────────────
try:
    from supabase import create_client as _sb_create
    import bcrypt as _bcrypt
    _sb = _sb_create("https://rvgmqmfmbknxxdyqpcgz.supabase.co","eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InJ2Z21xbWZtYmtueHhkeXFwY2d6Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3NzI2MzE0NTUsImV4cCI6MjA4ODIwNzQ1NX0.8J0rgNuwM0WGDduHQiuDG6PxcrrPP4h62Uh2U1Hp3nA")
    def _hash_pw(p): return _bcrypt.hashpw(p.encode(),_bcrypt.gensalt()).decode()
    def _verify_pw(p,h): return _bcrypt.checkpw(p.encode(),h.encode())
    def _signup(email,password,name,age):
        email=email.strip().lower()
        if _sb.table("users").select("id").eq("email",email).execute().data:
            return {"success":False,"error":"Email already registered."}
        try:
            r=_sb.table("users").insert({"email":email,"display_name":name,"age":age,"password_hash":_hash_pw(password)}).execute()
            uid=r.data[0]["id"]
            _sb.table("user_preferences").insert({"user_id":uid}).execute()
            return {"success":True,"user_id":uid,"display_name":name,"age":age,"prefs":{}}
        except Exception as e: return {"success":False,"error":str(e)}
    def _login(email,password):
        email=email.strip().lower()
        r=_sb.table("users").select("id,display_name,age,password_hash").eq("email",email).execute()
        if not r.data: return {"success":False,"error":"Email not found."}
        u=r.data[0]
        if not _verify_pw(password,u["password_hash"]): return {"success":False,"error":"Incorrect password."}
        prefs=_sb.table("user_preferences").select("*").eq("user_id",u["id"]).execute()
        return {"success":True,"user_id":u["id"],"display_name":u["display_name"],"age":u["age"],"prefs":prefs.data[0] if prefs.data else {}}
    def _save_month(uid,month,df_m):
        import json as _j
        try:
            _sb.table("user_statements").upsert({"user_id":uid,"month_period":month,"row_count":len(df_m),"data_json":_j.loads(df_m.to_json(orient="records",date_format="iso"))},on_conflict="user_id,month_period").execute()
            return True
        except: return False
    def _load_statements(uid):
        r=_sb.table("user_statements").select("data_json,month_period").eq("user_id",uid).order("month_period").execute()
        if not r.data: return pd.DataFrame()
        df=pd.concat([pd.DataFrame(row["data_json"]) for row in r.data],ignore_index=True)
        df["DATE"]=pd.to_datetime(df["DATE"]); return df
    def _delete_user(uid):
        try: _sb.table("users").delete().eq("id",uid).execute(); return True
        except: return False
    SUPABASE_OK = True
except Exception as _sbe:
    SUPABASE_OK = False
    def _signup(*a,**k): return {"success":False,"error":"Auth unavailable"}
    def _login(*a,**k): return {"success":False,"error":"Auth unavailable"}
    def _save_month(*a,**k): return False
    def _load_statements(*a,**k): return pd.DataFrame()
    def _delete_user(*a,**k): return False

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
    # Income & transfers
    if any(k in d for k in ["SALARY", "SAL ", "PAYROLL", "GIBL"]): return "Salary"
    elif any(k in d for k in ["TRF FROM", "TRANSFER IN", "IMPS/CR", "NEFT/CR", "CREDIT INTEREST"]): return "Transfer In"
    elif any(k in d for k in ["TRF TO", "TRANSFER OUT", "FUND TRANSFE"]): return "Transfer Out"
    # Digital payments
    elif "UPI" in d: return "UPI Payment"
    elif any(k in d for k in ["NEFT", "RTGS", "IMPS"]): return "Online Payment"
    # POS / Card swipes (MUST be before Shopping to catch generic card txns)
    elif any(k in d for k in ["POS ", "POS/", "POINT OF SALE", "CARD SWIPE", "ECOM",
        "CONTACTLESS", "DEBIT CARD", "MPS/"]): return "Card Payment"
    # Cash
    elif any(k in d for k in ["CASHDEP", "CASH DEP", "CASH DEPOSIT"]): return "Cash Deposit"
    elif any(k in d for k in ["ATM", "CDM", "CASHWDL", "CASH WDL", "CASH WITHDRAWAL"]): return "ATM/Cash Withdrawal"
    # Cheque
    elif any(k in d for k in ["CHQ", "CHEQUE", "CHECK", "CLG"]): return "Cheque"
    # Credit card bill payments
    elif any(k in d for k in ["CREDIT CARD", "CC BILL", "CC PAYMENT", "CARD BILL",
        "CRED ", "CRED/", "CARDPAY"]): return "Credit Card Bill"
    # Loans & EMI
    elif any(k in d for k in ["EMI", "LOAN", "MORTGAGE", "REPAYMENT"]): return "Loan/EMI"
    # Tax & Government
    elif any(k in d for k in ["TAX", "GST", "GOVT", "TDS", "INCOME TAX", "MCA", "EPFO", "PF"]): return "Tax/Government"
    # Shopping & E-commerce (includes fashion brands)
    elif any(k in d for k in ["AMAZON", "FLIPKART", "MYNTRA", "AJIO", "MEESHO", "SNAPDEAL",
        "SHOPPERS", "BIGBASKET", "BLINKIT", "ZEPTO", "JIOMART", "DMART", "GROFERS",
        "RELIANCE RETAIL", "TATA CLIQ", "NYKAA", "CROMA", "VIJAY SALES",
        "WESTSIDE", "TRENT", "PANTALOONS", "LIFESTYLE", "SHOPPERS STOP",
        "DECATHLON", "IKEA", "CHROMA", "RELIANCE DIGITAL", "LENSKART",
        "FIRSTCRY", "BEWAKOOF", "URBANIC", "H&M", "ZARA"]): return "Shopping"
    # Food & Dining
    elif any(k in d for k in ["SWIGGY", "ZOMATO", "DOMINOS", "MCDONALD", "KFC", "PIZZA",
        "STARBUCKS", "BURGER", "RESTAURANT", "FOOD", "CAFE", "DINING", "DUNZO",
        "EATSURE", "FAASOS", "HALDIRAM", "BARBEQUE"]): return "Food & Dining"
    # Travel & Transport
    elif any(k in d for k in ["UBER", "OLA", "RAPIDO", "IRCTC", "MAKEMYTRIP", "GOIBIBO",
        "CLEARTRIP", "REDBUS", "YATRA", "FLIGHT", "AIRLINE", "INDIGO", "SPICEJET",
        "RAILWAY", "METRO", "PETROL", "FUEL", "BPCL", "HPCL", "IOCL", "FASTAG",
        "PARKING", "TOLL"]): return "Travel & Transport"
    # Bills & Utilities
    elif any(k in d for k in ["ELECTRIC", "ELECTRICITY", "WATER BILL", "GAS BILL", "BESCOM",
        "TATA POWER", "ADANI", "BROADBAND", "INTERNET", "WIFI", "ACT FIBERNET",
        "AIRTEL", "JIO", "VODAFONE", "VI ", "BSNL", "MOBILE RECHARGE", "RECHARGE",
        "DTH", "TATA SKY", "DISH TV", "POSTPAID", "PREPAID"]): return "Bills & Utilities"
    # Insurance
    elif any(k in d for k in ["INSURANCE", "LIC ", "HDFC LIFE", "ICICI PRUDENTIAL",
        "SBI LIFE", "POLICY", "PREMIUM", "HEALTH INSURANCE", "STAR HEALTH",
        "MAX LIFE", "BAJAJ ALLIANZ"]): return "Insurance"
    # Subscriptions & Entertainment
    elif any(k in d for k in ["NETFLIX", "HOTSTAR", "PRIME VIDEO", "SPOTIFY", "YOUTUBE",
        "DISNEY", "SONY LIV", "ZEE5", "APPLE", "GOOGLE PLAY", "SUBSCRIPTION",
        "MEMBERSHIP", "GYM", "CULT FIT", "AUDIBLE", "BOOKMYSHOW", "BOOK MY SHOW",
        "PVR", "INOX", "CINEPOLIS", "MOVIE", "CINEMA"]): return "Entertainment"
    # Education
    elif any(k in d for k in ["SCHOOL", "COLLEGE", "UNIVERSITY", "TUITION", "COURSE",
        "UDEMY", "COURSERA", "UNACADEMY", "BYJU", "EDUCATION", "EXAM FEE",
        "UPGRAD"]): return "Education"
    # Rent & Housing (word-boundary safe — avoids matching "trentwestside")
    elif any(k in d for k in ["HOUSE RENT", "PG RENT", "MAINTENANCE", "SOCIETY"]) or \
         ((" RENT" in d or d.startswith("RENT") or "/RENT" in d) and "TRENT" not in d): return "Rent & Housing"
    # Medical & Health
    elif any(k in d for k in ["HOSPITAL", "PHARMACY", "MEDICAL", "DOCTOR", "CLINIC",
        "APOLLO", "MEDPLUS", "1MG", "PHARMEASY", "NETMEDS", "DIAGNOSTIC",
        "PATHLAB", "DENTAL"]): return "Medical & Health"
    # Investments
    elif any(k in d for k in ["MUTUAL FUND", "SIP ", "ZERODHA", "GROWW", "KUVERA",
        "DEMAT", "NSE ", "BSE ", "COIN ", "IPO",
        "SMALLCASE", "PPF", "NPS ", "FD ", "FIXED DEPOSIT", "RD "]): return "Investments"
    # Bank charges
    elif any(k in d for k in ["SERVICE CHARGE", "BANK CHARGE", "ANNUAL FEE", "LATE FEE",
        "PENALTY", "INTEREST CHARGED", "DEBIT INTEREST", "MIN BAL"]): return "Bank Charges"
    # Digital wallets & payment apps (non-UPI)
    elif any(k in d for k in ["PHONEPE", "PAYTM", "GPAY", "GOOGLE PAY", "MOBIKWIK",
        "FREECHARGE", "WALLET", "LAZYPAY", "SIMPL", "SLICE",
        "BHARATPE"]): return "Digital Wallet"
    # Payment gateways
    elif any(k in d for k in ["BILLDESK", "RAZORPAY", "PAYU", "CASHFREE", "CCAVENUE",
        "PAYGATE", "PAYMENT GATEWAY", "INSTAMOJO"]): return "Bill Payment"
    # Auto-debit / Standing instructions
    elif any(k in d for k in ["SI/", "ECS/", "ECS ", "NACH/", "NACH ", "AUTO DEBIT",
        "STANDING INSTRUCTION", "MANDATE", "E-MANDATE", "AUTOPAY"]): return "Auto-Debit"
    # Refunds & reversals
    elif any(k in d for k in ["REFUND", "REVERSAL", "CASHBACK", "REVERSL", "REV/",
        "FAILED TXN", "RETURN"]): return "Refund"
    # Dividend & interest income
    elif any(k in d for k in ["DIVIDEND", "DIV/", "INT ON", "INTEREST CREDIT",
        "INT.CREDIT", "BONUS"]): return "Dividend/Interest"
    # International / forex
    elif any(k in d for k in ["SWIFT", "FOREX", "WIRE TRANSFER", "FOREIGN",
        "INTERNATIONAL", "CROSS BORDER", "FCY"]): return "International"
    # Miscellaneous bank operations
    elif any(k in d for k in ["DEBIT MEMO", "ADJUSTMENT", "CLEARING", "SETTLEMENT",
        "SUSPENSE", "CONSOLIDATED", "MISC"]): return "Bank Misc"
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
    # ── Contextual Flagging Engine ──
    # Categories that are naturally spiky — need higher thresholds
    _HIGH_VAR_CATS = {"Travel & Transport", "UPI Payment", "Shopping",
                      "Card Payment", "Online Payment", "Bill Payment"}
    _cat_type_map = {}
    _sp = df[df["WITHDRAWAL AMT"]>0].copy()
    _sp["_MP"] = _sp["DATE"].dt.to_period("M")
    _tm = df["DATE"].dt.to_period("M").nunique()
    for _cat, _grp in _sp.groupby("CATEGORY"):
        _ms = _grp.groupby("_MP")["WITHDRAWAL AMT"].sum()
        _ar = _ms.shape[0] / _tm
        _cv = (_ms.std()/_ms.mean()) if _ms.std()>0 and _ms.mean()>0 else 0
        _rm = _ms.rolling(3,min_periods=1).mean()
        _cat_type_map[_cat] = {
            "type": "A" if _ar>=0.6 and _cv<0.25 else ("B" if _ar>=0.4 else "C"),
            "mm"  : float(_ms.mean()),
            "hmx" : float(_grp["WITHDRAWAL AMT"].max()),
            "rm"  : {str(k):float(v) for k,v in _rm.items()}
        }
    df["ALERT_LEVEL"]  = 0
    df["ALERT_REASON"] = ""
    _am = {}
    def _aa(_i,_lv,_rs):
        if _i not in _am or _lv>_am[_i][0]: _am[_i]=(_lv,_rs)
    for _cat,_pf in _cat_type_map.items():
        _cr=_sp[_sp["CATEGORY"]==_cat].copy()
        if _cr.empty: continue
        _ct=_pf["type"]; _mm=_pf["mm"]; _hmx=_pf["hmx"]; _rms=_pf["rm"]
        # Higher thresholds for naturally variable categories
        _is_hv = _cat in _HIGH_VAR_CATS
        if _ct=="A":
            _t_hard = 2.5 if _is_hv else 1.5
            _t_soft = 2.0 if _is_hv else 1.2
            for _i,_row in _cr.iterrows():
                _rv=_rms.get(str(_row["_MP"]),_mm) or _mm
                if _rv<=0: continue
                _rt=_row["WITHDRAWAL AMT"]/_rv
                if _rt>=_t_hard: _aa(_i,3,f"{_cat}: ₹{_row['WITHDRAWAL AMT']:,.0f} is {_rt:.1f}x expected ₹{_rv:,.0f} — significant spike.")
                elif _rt>=_t_soft: _aa(_i,2,f"{_cat}: ₹{_row['WITHDRAWAL AMT']:,.0f} is {_rt:.1f}x expected ₹{_rv:,.0f} — above normal.")
        elif _ct=="B":
            _t_hard = 4.0 if _is_hv else 2.9
            _t_soft = 3.0 if _is_hv else 1.9
            _cmo=_cr.groupby("_MP")["WITHDRAWAL AMT"].sum()
            for _mp,_mt in _cmo.items():
                _rv=_rms.get(str(_mp),_mm) or _mm
                if _rv<=0: continue
                _rt=_mt/_rv
                _sb=_cr[(_cr["_MP"]==_mp)&(_cr["WITHDRAWAL AMT"]>0)]
                if _sb.empty: continue
                _ai=_sb["WITHDRAWAL AMT"].idxmax()
                if _rt>_t_hard: _aa(_ai,3,f"{_cat}: Monthly ₹{_mt:,.0f} is {_rt:.1f}x rolling avg ₹{_rv:,.0f} — extreme spike.")
                elif _rt>_t_soft: _aa(_ai,2,f"{_cat}: Monthly ₹{_mt:,.0f} is {_rt:.1f}x rolling avg ₹{_rv:,.0f} — unusually high.")
        elif _ct=="C":
            _cs=_cr.sort_values("DATE")
            for _i,_row in _cs.iterrows():
                _pr=_cs[(_cs["DATE"]<_row["DATE"])&(_cs["DATE"]>=_row["DATE"]-pd.Timedelta(days=60))]
                _pc=len(_pr)
                if not _is_hv:
                    if _pc>=2: _aa(_i,3,f"{_cat}: {_pc+1} transactions in 60 days — unusually frequent.")
                    elif _pc==1: _aa(_i,2,f"{_cat}: 2nd transaction in {(_row['DATE']-_pr['DATE'].max()).days}d — typically rare.")
                if _hmx>0 and _row["WITHDRAWAL AMT"]>_hmx*1.5: _aa(_i,3,f"{_cat}: ₹{_row['WITHDRAWAL AMT']:,.0f} is 1.5x+ historical max ₹{_hmx:,.0f}.")
    for _i,(_lv,_rs) in _am.items():
        df.at[_i,"ALERT_LEVEL"]=_lv; df.at[_i,"ALERT_REASON"]=_rs
    df["IS_ANOMALY"] = (df["ALERT_LEVEL"]>0).astype(int)
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
    anomalies   = int(df["ALERT_LEVEL"].gt(0).sum())
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
    # Include actual flagged transactions so AI can reference them
    flagged = df[df["ALERT_LEVEL"] > 0].sort_values("ALERT_LEVEL", ascending=False)
    if len(flagged) > 0:
        flag_lines = []
        for _, frow in flagged.head(15).iterrows():
            lvl = {1:"Info",2:"Soft Alert",3:"Hard Alert"}.get(frow["ALERT_LEVEL"],"Flag")
            flag_lines.append(f"  - [{lvl}] {frow['DATE'].strftime('%d %b %Y')} | {str(frow['TRANSACTION DETAILS'])[:40]} | {cfmt(frow['WITHDRAWAL AMT'], None)} | Reason: {frow['ALERT_REASON']}")
        ctx += "Flagged Transactions (suspicious):\n" + "\n".join(flag_lines) + "\n"
    ctx += "Top Category: " + top_cat + "\n"
    ctx += "Monthly Trend:\n" + " | ".join(trend_parts) + "\n"
    ctx += "Top 5 Largest Withdrawals:\n" + "\n".join(top5_lines) + "\n"
    ctx += "Category Breakdown:\n" + "\n".join(cat_parts) + "\n"
    ctx += "IMPORTANT: Use ONLY this data. Always give specific numbers. Never say data unavailable. Be concise, friendly and actionable. When asked about anomalies or suspicious transactions, list the exact flagged transactions with their dates, amounts, and reasons."
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


# ── MicroRoundUp globals ──────────────────────────────────────
import urllib.request as _muu, json as _muj

@st.cache_data(ttl=86400)
def load_mu_databases():
    _b = "https://raw.githubusercontent.com/DarshMohapatra/FINSIGHT/main/"
    def _gj(f):
        try:
            with _muu.urlopen(_b+f, timeout=10) as resp: return _muj.load(resp)
        except Exception as e: st.warning(f"MU DB load failed: {e}"); return {}
    instr  = _gj("indian_instruments.json")
    xirr   = _gj("xirr_simulation.json")
    mc     = _gj("monte_carlo_projections.json")
    return instr, xirr, mc

MU_INSTR_RAW, MU_XIRR, MU_MC = load_mu_databases()
MU_INSTRUMENTS = MU_INSTR_RAW.get("instruments", []) if isinstance(MU_INSTR_RAW, dict) else []
MU_DISCLAIMER  = MU_INSTR_RAW.get("sebi_disclaimer", "") if isinstance(MU_INSTR_RAW, dict) else ""
MU_LAST_UPDATED= MU_INSTR_RAW.get("last_updated", "") if isinstance(MU_INSTR_RAW, dict) else ""

def mu_compute_roundups(df, threshold=10):
    df_w = df[df["WITHDRAWAL AMT"]>0].copy()
    remainder = df_w["WITHDRAWAL AMT"] % threshold
    df_w["ROUNDUP"] = threshold - remainder
    df_w.loc[remainder==0, "ROUNDUP"] = threshold
    monthly = (df_w.groupby(df_w["DATE"].dt.to_period("M"))
               .agg(roundup_total=("ROUNDUP","sum"),txn_count=("ROUNDUP","count"))
               .reset_index())
    monthly["DATE"] = monthly["DATE"].dt.to_timestamp()
    return df_w, monthly



# ── Year-in-Review: compute dynamically from uploaded data ────
import calendar as _yir_cal

def compute_yir_data(df):
    """Build Year-in-Review dict from the user's uploaded DataFrame."""
    if df is None or df.empty:
        return {}
    result = {}
    df = df.copy()
    df["_year"] = df["DATE"].dt.year
    for yr, gdf in df.groupby("_year"):
        spent     = float(gdf["WITHDRAWAL AMT"].sum())
        received  = float(gdf["DEPOSIT AMT"].sum())
        net_saved = received - spent
        total_txns = len(gdf)
        savings_rate = round(net_saved / received * 100, 1) if received > 0 else 0.0
        # Top merchant by amount
        wd = gdf[gdf["WITHDRAWAL AMT"] > 0]
        if not wd.empty:
            top_amt = wd.groupby("TRANSACTION DETAILS")["WITHDRAWAL AMT"].sum()
            top_merchant_name = top_amt.idxmax()
            top_merchant_amt  = float(top_amt.max())
        else:
            top_merchant_name, top_merchant_amt = "—", 0
        # Top merchant by frequency
        if not wd.empty:
            top_freq = wd["TRANSACTION DETAILS"].value_counts()
            top_freq_name  = top_freq.index[0]
            top_freq_count = int(top_freq.iloc[0])
        else:
            top_freq_name, top_freq_count = "—", 0
        # Most expensive day
        daily = gdf.groupby(gdf["DATE"].dt.date)["WITHDRAWAL AMT"].sum()
        if not daily.empty:
            exp_day  = str(daily.idxmax())
            exp_amt  = float(daily.max())
        else:
            exp_day, exp_amt = "—", 0
        # Biggest / quietest month
        gdf["_month_num"] = gdf["DATE"].dt.month
        monthly = gdf.groupby("_month_num")["WITHDRAWAL AMT"].sum()
        month_names = {m: _yir_cal.month_name[m] for m in range(1, 13)}
        # Fill missing months with 0
        all_months = pd.Series(0.0, index=range(1, 13))
        all_months.update(monthly)
        big_m  = int(all_months[all_months > 0].idxmax()) if (all_months > 0).any() else 1
        quiet_m = int(all_months[all_months > 0].idxmin()) if (all_months > 0).any() else 1
        # Top category
        if "CATEGORY" in gdf.columns and not wd.empty:
            cat_spend = wd.groupby("CATEGORY")["WITHDRAWAL AMT"].sum().sort_values(ascending=False)
            top_cat_name = cat_spend.index[0] if len(cat_spend) > 0 else "—"
            top_cat_amt  = float(cat_spend.iloc[0]) if len(cat_spend) > 0 else 0
            top_cat_pct  = round(top_cat_amt / spent * 100, 1) if spent > 0 else 0
        else:
            top_cat_name, top_cat_amt, top_cat_pct = "—", 0, 0
        # Largest single transaction
        if not wd.empty:
            largest_idx  = wd["WITHDRAWAL AMT"].idxmax()
            largest_amt  = float(wd.loc[largest_idx, "WITHDRAWAL AMT"])
            largest_desc = str(wd.loc[largest_idx, "TRANSACTION DETAILS"])[:25]
        else:
            largest_amt, largest_desc = 0, "—"
        # Anomaly count
        anomaly_count = int(gdf["ALERT_LEVEL"].gt(0).sum()) if "ALERT_LEVEL" in gdf.columns else 0
        # Monthly spend dict
        monthly_spend = {}
        for m in range(1, 13):
            monthly_spend[month_names[m]] = float(all_months.get(m, 0))
        result[str(yr)] = {
            "year": int(yr),
            "total_txns": total_txns,
            "total_spent": spent,
            "total_received": received,
            "net_saved": net_saved,
            "savings_rate": savings_rate,
            "top_merchant_by_amount": {"name": top_merchant_name, "amount": top_merchant_amt},
            "top_merchant_by_frequency": {"name": top_freq_name, "count": top_freq_count},
            "most_expensive_day": {"date": exp_day, "amount": exp_amt},
            "biggest_month": {"name": month_names[big_m], "amount": float(all_months[big_m])},
            "quietest_month": {"name": month_names[quiet_m], "amount": float(all_months[quiet_m])},
            "top_category": {"name": top_cat_name, "pct": top_cat_pct},
            "largest_txn": {"amount": largest_amt, "desc": largest_desc},
            "anomaly_count": anomaly_count,
            "monthly_spend": monthly_spend
        }
    return result

if st.session_state.page == "landing":
    components.html(LANDING_HTML, height=2400, scrolling=True)
    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("⚡ Launch App — Analyze My Finances", use_container_width=True):
            st.session_state.page = "auth"
            st.rerun()
elif st.session_state.page == "auth":
    _ac1,_ac2,_ac3 = st.columns([1,2,1])
    with _ac2:
        st.markdown("<br><br>", unsafe_allow_html=True)
        st.markdown('<div style="text-align:center;margin-bottom:32px"><span style="font-size:36px;font-weight:900">⚡<span style="background:linear-gradient(135deg,#00f5a0,#00d4ff);-webkit-background-clip:text;-webkit-text-fill-color:transparent">FinSight</span></span><br><span style="font-family:DM Mono,monospace;font-size:10px;color:rgba(255,255,255,0.3);letter-spacing:3px">AI FINANCIAL INTELLIGENCE</span></div>', unsafe_allow_html=True)
        _t1,_t2 = st.tabs(["🔑  LOGIN","✨  SIGN UP"])
        with _t1:
            st.markdown("<br>", unsafe_allow_html=True)
            _le = st.text_input("Email", placeholder="you@example.com", key="login_email")
            _lp = st.text_input("Password", type="password", placeholder="••••••••", key="login_pass")
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("⚡ Login to FinSight", use_container_width=True, key="btn_login"):
                if _le and _lp:
                    with st.spinner("Verifying..."):
                        _res = _login(_le, _lp)
                    if _res["success"]:
                        st.session_state.auth_user = _res
                        with st.spinner("Loading your data..."):
                            _stored = _load_statements(_res["user_id"])
                        if not _stored.empty:
                            st.session_state.user_df = _stored
                            st.session_state.analysis_done = True
                            st.session_state.forecast_cache = None
                            st.session_state.forecast_df_id = None
                        st.session_state.page = "app"
                        st.rerun()
                    else:
                        st.error("❌ " + _res["error"])
                else:
                    st.warning("Please enter email and password.")
        with _t2:
            st.markdown("<br>", unsafe_allow_html=True)
            _sn  = st.text_input("Full Name", placeholder="Your name", key="signup_name")
            _se  = st.text_input("Email", placeholder="you@example.com", key="signup_email")
            _sa  = st.number_input("Age", min_value=16, max_value=100, value=25, key="signup_age")
            _sp  = st.text_input("Password", type="password", placeholder="Min 8 characters", key="signup_pass")
            _sp2 = st.text_input("Confirm Password", type="password", placeholder="••••••••", key="signup_pass2")
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("✨ Create Account", use_container_width=True, key="btn_signup"):
                if not all([_sn,_se,_sp,_sp2]):
                    st.warning("Please fill all fields.")
                elif _sp != _sp2:
                    st.error("Passwords do not match.")
                elif len(_sp) < 8:
                    st.error("Password must be at least 8 characters.")
                else:
                    with st.spinner("Creating account..."):
                        _res = _signup(_se,_sp,_sn,int(_sa))
                    if _res["success"]:
                        st.session_state.auth_user = _res
                        st.session_state.page = "app"
                        st.rerun()
                    else:
                        st.error("❌ " + _res["error"])
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("← Back to Home", key="btn_auth_back"):
            st.session_state.page = "landing"
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

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["📤  UPLOAD", "📊  DASHBOARD", "📈  FORECAST", "🤖  AI ADVISOR", "💳  SMARTCASH", "💰  INVEST"])

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
                            if st.session_state.get("auth_user"):
                                _uid=st.session_state.auth_user["user_id"]
                                for _mp,_mdf in df.groupby(df["DATE"].dt.to_period("M")):
                                    _save_month(_uid,str(_mp),_mdf)
                        except Exception as e:
                            st.error("❌ " + str(e))
        with c2:
            st.markdown('<div class="glass" style="margin-top:72px;"><div style="font-family:DM Mono,monospace;font-size:10px;color:#00f5a0;letter-spacing:2px;margin-bottom:16px;">AUTO-DETECTED COLUMNS</div><table style="width:100%;border-collapse:collapse;font-size:12px;"><tr style="color:rgba(255,255,255,0.3);"><td style="padding:6px 0;font-family:DM Mono,monospace;">Required As</td><td style="padding:6px 0;font-family:DM Mono,monospace;">Also Accepted</td></tr><tr style="border-top:1px solid rgba(255,255,255,0.06);"><td style="padding:8px 0;color:#00f5a0;font-family:DM Mono,monospace;font-size:11px;">DATE</td><td style="padding:8px 0;color:rgba(255,255,255,0.4);font-size:11px;">Transaction Date, Timestamp</td></tr><tr style="border-top:1px solid rgba(255,255,255,0.06);"><td style="padding:8px 0;color:#00f5a0;font-family:DM Mono,monospace;font-size:11px;">TRANSACTION DETAILS</td><td style="padding:8px 0;color:rgba(255,255,255,0.4);font-size:11px;">Narration, Type, Category</td></tr><tr style="border-top:1px solid rgba(255,255,255,0.06);"><td style="padding:8px 0;color:#00f5a0;font-family:DM Mono,monospace;font-size:11px;">WITHDRAWAL AMT</td><td style="padding:8px 0;color:rgba(255,255,255,0.4);font-size:11px;">Debit, Debit Amt, DR</td></tr><tr style="border-top:1px solid rgba(255,255,255,0.06);"><td style="padding:8px 0;color:rgba(255,255,255,0.5);font-family:DM Mono,monospace;font-size:11px;">DEPOSIT AMT</td><td style="padding:8px 0;color:rgba(255,255,255,0.4);font-size:11px;">Credit, Credit Amt, CR</td></tr><tr style="border-top:1px solid rgba(255,255,255,0.06);"><td style="padding:8px 0;color:rgba(255,255,255,0.5);font-family:DM Mono,monospace;font-size:11px;">BALANCE AMT</td><td style="padding:8px 0;color:rgba(255,255,255,0.4);font-size:11px;">Balance, Closing Balance</td></tr></table></div>', unsafe_allow_html=True)

        if st.session_state.analysis_done and st.session_state.user_df is not None:
            df = st.session_state.user_df
            st.markdown("<hr class='divider'>", unsafe_allow_html=True)
            # ── Guardrails FIRST (so anomaly count is correct) ──
            st.markdown('<div style="margin:20px 0 4px;padding:20px 24px;background:linear-gradient(145deg,rgba(245,158,11,0.06),rgba(245,158,11,0.02));border:1px solid rgba(245,158,11,0.15);border-radius:14px"><div style="display:flex;align-items:center;gap:8px;margin-bottom:14px"><span style="font-size:16px">🔍</span><span style="font-family:DM Mono,monospace;font-size:11px;color:#f59e0b;letter-spacing:2px;font-weight:600">SET YOUR GUARDRAILS</span><span style="font-size:10px;color:rgba(255,255,255,0.25);margin-left:auto">Set limits → click Scan</span></div>', unsafe_allow_html=True)
            _wd_vals = df[df["WITHDRAWAL AMT"] > 0]["WITHDRAWAL AMT"]
            _default_txn = str(int(round(_wd_vals.quantile(0.95) / 100) * 100)) if len(_wd_vals) > 10 else "10000"
            _monthly_spend = df.groupby(df["DATE"].dt.to_period("M"))["WITHDRAWAL AMT"].sum()
            _default_monthly = str(int(round(_monthly_spend.quantile(0.85) / 1000) * 1000)) if len(_monthly_spend) > 2 else "50000"
            # Initialize session defaults only once
            if "guard_txn_val" not in st.session_state:
                st.session_state["guard_txn_val"] = _default_txn
            if "guard_monthly_val" not in st.session_state:
                st.session_state["guard_monthly_val"] = _default_monthly
            _gc1, _gc2, _gc3 = st.columns([1, 1, 1])
            with _gc1:
                st.markdown('<div style="font-size:10px;color:rgba(255,255,255,0.4);font-family:DM Mono,monospace;letter-spacing:1px;margin-bottom:4px">MAX PER TRANSACTION (₹)</div>', unsafe_allow_html=True)
                _guard_txn_str = st.text_input("txn_limit", value=st.session_state["guard_txn_val"], key="guard_txn_input", label_visibility="collapsed", placeholder="e.g. 10000")
            with _gc2:
                st.markdown('<div style="font-size:10px;color:rgba(255,255,255,0.4);font-family:DM Mono,monospace;letter-spacing:1px;margin-bottom:4px">MONTHLY BUDGET LIMIT (₹)</div>', unsafe_allow_html=True)
                _guard_monthly_str = st.text_input("monthly_limit", value=st.session_state["guard_monthly_val"], key="guard_monthly_input", label_visibility="collapsed", placeholder="e.g. 50000")
            with _gc3:
                st.markdown('<div style="font-size:10px;color:rgba(255,255,255,0.4);font-family:DM Mono,monospace;letter-spacing:1px;margin-bottom:4px">WATCH CATEGORIES</div>', unsafe_allow_html=True)
                _guard_cats = st.multiselect("cats", options=sorted(df["CATEGORY"].unique()), default=[], key="guard_cats", label_visibility="collapsed")
            st.markdown('</div>', unsafe_allow_html=True)
            # ── Scan Button ──
            _scan_col1, _scan_col2, _scan_col3 = st.columns([1, 1, 1])
            with _scan_col2:
                _scan_clicked = st.button("🔍  Scan Transactions", key="scan_btn", use_container_width=True, type="primary")
            if _scan_clicked:
                st.session_state["guard_txn_val"] = _guard_txn_str
                st.session_state["guard_monthly_val"] = _guard_monthly_str
                st.session_state["guardrails_applied"] = True
            # Parse text inputs safely
            _active_txn_str = st.session_state.get("guard_txn_val", _default_txn)
            _active_monthly_str = st.session_state.get("guard_monthly_val", _default_monthly)
            try: _guard_txn = int(str(_active_txn_str).replace(",","").replace("₹","").strip())
            except: _guard_txn = 10000
            try: _guard_monthly = int(str(_active_monthly_str).replace(",","").replace("₹","").strip())
            except: _guard_monthly = 50000
            # Reset guardrail-level flags before re-applying (keep contextual flags)
            _ctx_mask = df["ALERT_LEVEL"] == 3  # preserve hard alerts from contextual engine
            df.loc[~_ctx_mask, "ALERT_LEVEL"] = 0
            df.loc[~_ctx_mask, "ALERT_REASON"] = ""
            # Apply user guardrails on top of contextual engine
            for _gi, _grow in df.iterrows():
                _reasons = []
                if _grow["WITHDRAWAL AMT"] >= _guard_txn:
                    _reasons.append(f"₹{_grow['WITHDRAWAL AMT']:,.0f} exceeds your ₹{_guard_txn:,.0f} per-txn limit")
                if _guard_cats and _grow["CATEGORY"] in _guard_cats and _grow["WITHDRAWAL AMT"] > 0:
                    _reasons.append(f"'{_grow['CATEGORY']}' is on your watch list")
                if _reasons:
                    if df.at[_gi, "ALERT_LEVEL"] < 2:
                        df.at[_gi, "ALERT_LEVEL"] = 2
                        df.at[_gi, "ALERT_REASON"] = " | ".join(_reasons)
            _monthly_totals = df.groupby(df["DATE"].dt.to_period("M"))["WITHDRAWAL AMT"].sum()
            _over_budget = _monthly_totals[_monthly_totals > _guard_monthly]
            for _obm in _over_budget.index:
                _ob_rows = df[(df["DATE"].dt.to_period("M") == _obm) & (df["WITHDRAWAL AMT"] > 0)]
                if not _ob_rows.empty:
                    _top_idx = _ob_rows["WITHDRAWAL AMT"].idxmax()
                    if df.at[_top_idx, "ALERT_LEVEL"] < 1:
                        df.at[_top_idx, "ALERT_LEVEL"] = 1
                        df.at[_top_idx, "ALERT_REASON"] = f"Month total ₹{_monthly_totals[_obm]:,.0f} exceeds your ₹{_guard_monthly:,.0f} budget"
            df["IS_ANOMALY"] = (df["ALERT_LEVEL"] > 0).astype(int)
            st.session_state.user_df = df
            # ── Metrics (NOW correct since guardrails already applied) ──
            _anomaly_count = int(df["ALERT_LEVEL"].gt(0).sum())
            st.markdown('<div style="font-family:DM Mono,monospace;font-size:11px;color:#00f5a0;letter-spacing:2px;margin:20px 0 16px">ANALYSIS SUMMARY</div>', unsafe_allow_html=True)
            c1, c2, c3, c4 = st.columns(4)
            _anom_color = "#ff4d6d" if _anomaly_count > 0 else "#00f5a0"
            for col, (val, lbl, clr) in zip([c1, c2, c3, c4], [
                (str(len(df)), "TRANSACTIONS", "#00f5a0"),
                (cfmt(df["WITHDRAWAL AMT"].sum(), None), "TOTAL WITHDRAWN", "#00d4ff"),
                (str(_anomaly_count), "ANOMALIES FLAGGED", _anom_color),
                (str(round((df["CATEGORY"] != "Other").sum()/len(df)*100, 1)) + "%", "AUTO CATEGORIZED", "#a855f7"),
            ]):
                with col:
                    st.markdown(f'<div style="padding:18px;background:linear-gradient(145deg,rgba(255,255,255,0.03),rgba(255,255,255,0.01));border:1px solid rgba(255,255,255,0.08);border-radius:14px;text-align:center"><div style="font-size:28px;font-weight:800;color:{clr};letter-spacing:-0.5px">{val}</div><div style="font-size:9px;color:rgba(255,255,255,0.3);font-family:DM Mono,monospace;letter-spacing:1.5px;margin-top:6px">{lbl}</div></div>', unsafe_allow_html=True)
            # ── Flagged Transactions ──
            _flagged = df[df["ALERT_LEVEL"] > 0].sort_values("ALERT_LEVEL", ascending=False)
            _total_flagged = len(_flagged)
            if _total_flagged > 0:
                st.markdown(f'<div style="margin:20px 0 12px;padding:16px 20px;background:linear-gradient(145deg,rgba(255,77,109,0.08),rgba(255,77,109,0.02));border:1px solid rgba(255,77,109,0.2);border-radius:14px"><div style="display:flex;align-items:center;gap:8px;margin-bottom:12px"><span style="font-size:14px">🚨</span><span style="font-family:DM Mono,monospace;font-size:11px;color:#ff4d6d;letter-spacing:2px;font-weight:600">FLAGGED TRANSACTIONS — {_total_flagged} SUSPICIOUS</span></div>', unsafe_allow_html=True)
                _fdisp = _flagged[["DATE", "TRANSACTION DETAILS", "WITHDRAWAL AMT", "CATEGORY", "ALERT_LEVEL", "ALERT_REASON"]].copy()
                _fdisp["ALERT_LEVEL"] = _fdisp["ALERT_LEVEL"].map({0:"✅ Clean",1:"🔵 Budget",2:"🟡 Guardrail",3:"🔴 Hard Alert"})
                _fdisp["WITHDRAWAL AMT"] = _fdisp["WITHDRAWAL AMT"].apply(lambda x: cfmt(x, None))
                st.dataframe(_fdisp, use_container_width=True, hide_index=True)
                st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div style="margin:20px 0 12px;padding:16px 20px;background:rgba(0,245,160,0.04);border:1px solid rgba(0,245,160,0.12);border-radius:14px;display:flex;align-items:center;gap:10px"><span style="font-size:14px">✅</span><span style="font-family:DM Mono,monospace;font-size:11px;color:#00f5a0">All clear — no suspicious transactions. Lower your guardrail limits above to flag more.</span></div>', unsafe_allow_html=True)
            st.markdown('<div style="font-family:DM Mono,monospace;font-size:11px;color:#00f5a0;letter-spacing:2px;margin:20px 0 12px">RECENT TRANSACTIONS</div>', unsafe_allow_html=True)
            disp = df[["DATE", "TRANSACTION DETAILS", "WITHDRAWAL AMT", "DEPOSIT AMT", "CATEGORY", "ALERT_LEVEL"]].head(20).copy()
            disp["ALERT_LEVEL"] = disp["ALERT_LEVEL"].map({0:"✅ Clean",1:"🔵 Budget",2:"🟡 Guardrail",3:"🔴 Hard"})
            disp["WITHDRAWAL AMT"] = disp["WITHDRAWAL AMT"].apply(lambda x: cfmt(x, None))
            disp["DEPOSIT AMT"]    = disp["DEPOSIT AMT"].apply(lambda x: cfmt(x, None))
            st.dataframe(disp, use_container_width=True, hide_index=True)

    with tab2:
        # ── Year-in-Review (computed from uploaded data) ──────────
        YIR_DATA = compute_yir_data(st.session_state.get("user_df"))
        _yir_years = sorted([int(k) for k in YIR_DATA.keys()], reverse=True) if YIR_DATA else []
        if _yir_years:
            _yir_c1, _yir_c2 = st.columns([3,1])
            _yir_c1.markdown('<div style="font-family:DM Mono,monospace;font-size:10px;color:#00f5a0;letter-spacing:3px;margin-bottom:4px">✨ YEAR IN REVIEW</div>', unsafe_allow_html=True)
            _yir_sel = _yir_c2.selectbox("", _yir_years, index=0, key="yir_year", label_visibility="collapsed")
            _yd = YIR_DATA.get(str(_yir_sel), {})
            if _yd:
                st.markdown(f'<div style="margin:0 0 20px;padding:24px 32px;background:linear-gradient(135deg,rgba(0,245,160,0.08),rgba(0,212,255,0.04));border:1px solid rgba(0,245,160,0.2);border-radius:16px;text-align:center"><div style="font-family:DM Mono,monospace;font-size:10px;color:#00f5a0;letter-spacing:3px;margin-bottom:8px">{_yir_sel} · YEAR IN REVIEW</div><div style="font-size:42px;font-weight:900;margin-bottom:4px">{cfmt(_yd.get("total_spent",0), None)}</div><div style="color:rgba(255,255,255,0.4);font-size:13px">total spent across {_yd.get("total_txns",0):,} transactions</div><div style="margin-top:12px;font-size:16px;color:{"#00f5a0" if _yd.get("net_saved",0)>=0 else "#ff3c64"};font-weight:700">{"💰 Saved " if _yd.get("net_saved",0)>=0 else "📉 Deficit "}{cfmt(abs(_yd.get("net_saved",0)), None)} · {_yd.get("savings_rate",0)}% savings rate</div></div>', unsafe_allow_html=True)
                _yr1, _yr2, _yr3, _yr4 = st.columns(4)
                _yr1.markdown(f'<div style="padding:16px;background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.07);border-radius:12px;margin-bottom:10px"><div style="font-size:9px;color:rgba(255,255,255,0.35);font-family:DM Mono,monospace;letter-spacing:2px;margin-bottom:6px">🏆 TOP MERCHANT</div><div style="font-size:15px;font-weight:700;margin-bottom:2px">{_yd.get("top_merchant_by_amount",{}).get("name","—")[:22]}</div><div style="font-size:11px;color:#00f5a0">{cfmt(_yd.get("top_merchant_by_amount",{}).get("amount",0), None)}</div></div>', unsafe_allow_html=True)
                _yr2.markdown(f'<div style="padding:16px;background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.07);border-radius:12px;margin-bottom:10px"><div style="font-size:9px;color:rgba(255,255,255,0.35);font-family:DM Mono,monospace;letter-spacing:2px;margin-bottom:6px">📅 BIGGEST DAY</div><div style="font-size:15px;font-weight:700;margin-bottom:2px">{_yd.get("most_expensive_day",{}).get("date","—")}</div><div style="font-size:11px;color:#00f5a0">{cfmt(_yd.get("most_expensive_day",{}).get("amount",0), None)}</div></div>', unsafe_allow_html=True)
                _yr3.markdown(f'<div style="padding:16px;background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.07);border-radius:12px;margin-bottom:10px"><div style="font-size:9px;color:rgba(255,255,255,0.35);font-family:DM Mono,monospace;letter-spacing:2px;margin-bottom:6px">🔥 BIGGEST MONTH</div><div style="font-size:15px;font-weight:700;margin-bottom:2px">{_yd.get("biggest_month",{}).get("name","—")}</div><div style="font-size:11px;color:#00f5a0">{cfmt(_yd.get("biggest_month",{}).get("amount",0), None)}</div></div>', unsafe_allow_html=True)
                _yr4.markdown(f'<div style="padding:16px;background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.07);border-radius:12px;margin-bottom:10px"><div style="font-size:9px;color:rgba(255,255,255,0.35);font-family:DM Mono,monospace;letter-spacing:2px;margin-bottom:6px">😴 QUIETEST MONTH</div><div style="font-size:15px;font-weight:700;margin-bottom:2px">{_yd.get("quietest_month",{}).get("name","—")}</div><div style="font-size:11px;color:#a78bfa">{cfmt(_yd.get("quietest_month",{}).get("amount",0), None)}</div></div>', unsafe_allow_html=True)
                _yr5, _yr6, _yr7, _yr8 = st.columns(4)
                _yr5.markdown(f'<div style="padding:16px;background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.07);border-radius:12px;margin-bottom:10px"><div style="font-size:9px;color:rgba(255,255,255,0.35);font-family:DM Mono,monospace;letter-spacing:2px;margin-bottom:6px">🍕 TOP CATEGORY</div><div style="font-size:15px;font-weight:700;margin-bottom:2px">{_yd.get("top_category",{}).get("name","—")[:18]}</div><div style="font-size:11px;color:#f59e0b">{_yd.get("top_category",{}).get("pct",0)}% of spend</div></div>', unsafe_allow_html=True)
                _yr6.markdown(f'<div style="padding:16px;background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.07);border-radius:12px;margin-bottom:10px"><div style="font-size:9px;color:rgba(255,255,255,0.35);font-family:DM Mono,monospace;letter-spacing:2px;margin-bottom:6px">⚡ LARGEST TXN</div><div style="font-size:15px;font-weight:700;margin-bottom:2px">{cfmt(_yd.get("largest_txn",{}).get("amount",0), None)}</div><div style="font-size:11px;color:rgba(255,255,255,0.4)">{_yd.get("largest_txn",{}).get("desc","—")[:20]}</div></div>', unsafe_allow_html=True)
                _yr7.markdown(f'<div style="padding:16px;background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.07);border-radius:12px;margin-bottom:10px"><div style="font-size:9px;color:rgba(255,255,255,0.35);font-family:DM Mono,monospace;letter-spacing:2px;margin-bottom:6px">🔄 MOST FREQUENT</div><div style="font-size:15px;font-weight:700;margin-bottom:2px">{_yd.get("top_merchant_by_frequency",{}).get("name","—")[:22]}</div><div style="font-size:11px;color:#00d4ff">{_yd.get("top_merchant_by_frequency",{}).get("count",0)} times</div></div>', unsafe_allow_html=True)
                _yr8.markdown(f'<div style="padding:16px;background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.07);border-radius:12px;margin-bottom:10px"><div style="font-size:9px;color:rgba(255,255,255,0.35);font-family:DM Mono,monospace;letter-spacing:2px;margin-bottom:6px">🚨 ANOMALIES</div><div style="font-size:15px;font-weight:700;margin-bottom:2px">{_yd.get("anomaly_count",0):,}</div><div style="font-size:11px;color:#ff3c64">suspicious txns</div></div>', unsafe_allow_html=True)
                _yir_monthly = _yd.get("monthly_spend", {})
                if _yir_monthly:
                    import plotly.graph_objects as _yirgo
                    _yir_months = list(_yir_monthly.keys())
                    _yir_vals   = list(_yir_monthly.values())
                    _yir_fig = _yirgo.Figure()
                    _yir_fig.add_trace(_yirgo.Bar(x=_yir_months,y=_yir_vals,
                        marker_color=["#00f5a0" if v==max(_yir_vals) else "#1a3a2a" for v in _yir_vals],
                        hovertemplate="<b>%{x}</b><br>₹%{y:,.0f}<extra></extra>"))
                    _yir_fig.update_layout(paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",
                        margin=dict(l=0,r=0,t=0,b=0),height=140,
                        xaxis=dict(showgrid=False,color="#444",tickfont=dict(size=9)),
                        yaxis=dict(showgrid=False,visible=False),showlegend=False)
                    st.markdown('<div style="font-size:9px;color:rgba(255,255,255,0.3);font-family:DM Mono,monospace;letter-spacing:2px;margin-bottom:4px">MONTHLY SPEND PATTERN</div>', unsafe_allow_html=True)
                    st.plotly_chart(_yir_fig, use_container_width=True)
        st.markdown('<div style="border-top:1px solid rgba(255,255,255,0.06);margin:8px 0 24px"></div>', unsafe_allow_html=True)

        if st.session_state.user_df is None:
            st.markdown('<div style="text-align:center;padding:100px 0;"><div style="font-size:56px;margin-bottom:16px;">📤</div><div style="font-family:DM Mono,monospace;font-size:12px;color:rgba(255,255,255,0.2);letter-spacing:2px;">UPLOAD A FILE FIRST</div></div>', unsafe_allow_html=True)
        else:
            df_full = st.session_state.user_df
            # Year filter tied to Year-in-Review
            _dash_years = sorted(df_full["DATE"].dt.year.dropna().unique(), reverse=True)
            _dc1, _dc2 = st.columns([3,1])
            _dc1.markdown('<div style="padding:28px 0 0;font-family:DM Mono,monospace;font-size:11px;color:#00f5a0;letter-spacing:2px;margin-bottom:20px;">YOUR FINANCIAL DASHBOARD</div>', unsafe_allow_html=True)
            _dash_opt = ["All Years"] + [str(y) for y in _dash_years]
            _dash_sel = _dc2.selectbox("", _dash_opt, index=0, key="dash_year", label_visibility="collapsed")
            if _dash_sel != "All Years":
                df = df_full[df_full["DATE"].dt.year == int(_dash_sel)].copy()
            else:
                df = df_full
            fig, axes = dark_fig(2, 2, (16, 10))
            _dash_title = f"FinSight Financial Dashboard — {_dash_sel}" if _dash_sel != "All Years" else "FinSight Financial Dashboard"
            fig.suptitle(_dash_title, fontsize=16, fontweight="bold", color="white", y=0.98)
            pal = ["#00f5a0","#00d4ff","#7b61ff","#ff4d6d","#ffd60a","#ff9f43","#a8e063","#f8a5c2","#778ca3","#2d98da","#4b7bec","#a55eea","#e056fd","#c7ecee","#dfe6e9","#fd79a8"]
            mwd = df.groupby(df["DATE"].dt.to_period("M"))["WITHDRAWAL AMT"].sum()
            axes[0,0].plot(range(len(mwd)), mwd.values, color="#00f5a0", linewidth=2.5)
            axes[0,0].fill_between(range(len(mwd)), mwd.values, alpha=0.1, color="#00f5a0")
            axes[0,0].set_title("Monthly Withdrawals", color="white", fontsize=12, pad=12)
            axes[0,0].yaxis.set_major_formatter(mticker.FuncFormatter(cfmt))
            step = max(1, len(mwd)//8)
            axes[0,0].set_xticks(range(0, len(mwd), step))
            axes[0,0].set_xticklabels(mwd.index.astype(str)[::step], rotation=45, ha="right", color="#555", fontsize=8)
            # Category by spend amount (not count) — excludes categories with 0 spend
            cat_spend = df[df["WITHDRAWAL AMT"] > 0].groupby("CATEGORY")["WITHDRAWAL AMT"].sum().sort_values(ascending=True)
            cat_spend = cat_spend[cat_spend > 0]
            axes[0,1].barh(cat_spend.index, cat_spend.values, color=pal[:len(cat_spend)])
            axes[0,1].set_title("Spending by Category", color="white", fontsize=12, pad=12)
            axes[0,1].tick_params(colors="#aaa", labelsize=9)
            axes[0,1].xaxis.set_major_formatter(mticker.FuncFormatter(cfmt))
            am = df[df["ALERT_LEVEL"]>0].groupby("MONTH")["IS_ANOMALY"].count()
            x = list(range(len(am)))
            if x:
                axes[1,0].bar(x, am.values, color="#ff4d6d", alpha=0.85, edgecolor="none")
                axes[1,0].set_title("Anomalies Per Month", color="white", fontsize=12, pad=12)
                step2 = max(1, len(x)//8)
                axes[1,0].set_xticks(x[::step2])
                axes[1,0].set_xticklabels(am.index.astype(str)[::step2], rotation=45, ha="right", color="#555", fontsize=8)
            else:
                axes[1,0].set_title("Anomalies Per Month", color="white", fontsize=12, pad=12)
                axes[1,0].text(0.5, 0.5, "No anomalies", ha="center", va="center", color="#555", fontsize=12, transform=axes[1,0].transAxes)
            wd = df[df["WITHDRAWAL AMT"] > 0]["WITHDRAWAL AMT"]
            if len(wd) > 0:
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
        st.markdown('<div style="padding:32px 40px 0"><div style="font-family:DM Mono,monospace;font-size:10px;color:#00f5a0;letter-spacing:3px;margin-bottom:8px">FEATURE 2 — MICROROUNDUP</div><div style="font-size:26px;font-weight:800;margin-bottom:8px">💰 MicroRoundUp India</div><div style="color:rgba(255,255,255,0.4);font-size:14px;margin-bottom:24px">Round up every transaction to the nearest ₹10/₹50/₹100. Invest the spare change into Nifty ETF, Gold, or ELSS. Watch it compound.</div></div>', unsafe_allow_html=True)
        if st.session_state.get("user_df") is None:
            st.info("Upload your bank statement in the UPLOAD tab first.")
        else:
            df_mu = st.session_state["user_df"]
            # --- Threshold selector (hidden radio for state) ---
            _thr_cols = st.columns([1,1,1,3])
            threshold = 10  # default
            if "mu_threshold" not in st.session_state:
                st.session_state["mu_threshold"] = 10
            with _thr_cols[0]:
                if st.button("₹10", key="mu_btn10", use_container_width=True):
                    st.session_state["mu_threshold"] = 10
            with _thr_cols[1]:
                if st.button("₹50", key="mu_btn50", use_container_width=True):
                    st.session_state["mu_threshold"] = 50
            with _thr_cols[2]:
                if st.button("₹100", key="mu_btn100", use_container_width=True):
                    st.session_state["mu_threshold"] = 100
            threshold = st.session_state["mu_threshold"]
            # Style the active button
            active_btn_css = f"""<style>
            div[data-testid="stHorizontalBlock"] button {{
                background: rgba(255,255,255,0.04) !important;
                border: 1px solid rgba(255,255,255,0.1) !important;
                color: rgba(255,255,255,0.5) !important;
                border-radius: 10px !important;
                font-family: 'DM Mono', monospace !important;
                font-size: 13px !important;
                font-weight: 600 !important;
                padding: 8px 0 !important;
                transition: all 0.2s !important;
            }}
            div[data-testid="stHorizontalBlock"] button:hover {{
                border-color: #00f5a0 !important;
                color: #00f5a0 !important;
            }}
            </style>"""
            st.markdown(active_btn_css, unsafe_allow_html=True)
            st.markdown(f'<div style="font-family:DM Mono,monospace;font-size:10px;color:rgba(255,255,255,0.35);letter-spacing:1px;margin:-8px 0 16px 4px">ROUND-UP THRESHOLD · <span style="color:#00f5a0;font-weight:700">₹{threshold}</span> PER TRANSACTION</div>', unsafe_allow_html=True)
            df_rw, monthly_ru = mu_compute_roundups(df_mu, threshold)
            total_corpus  = float(df_rw["ROUNDUP"].sum())
            monthly_avg   = total_corpus / max(len(monthly_ru), 1)
            total_txns    = len(df_rw)
            avg_per_txn   = total_corpus / max(total_txns, 1)
            # --- Styled metric cards ---
            st.markdown(f'''<div style="display:grid;grid-template-columns:1fr 1fr 1fr 1fr;gap:14px;margin-bottom:24px">
<div style="padding:20px;background:linear-gradient(145deg,rgba(0,245,160,0.08),rgba(0,245,160,0.02));border:1px solid rgba(0,245,160,0.15);border-radius:14px">
<div style="font-size:10px;color:rgba(255,255,255,0.35);font-family:DM Mono,monospace;letter-spacing:1.5px;margin-bottom:8px">TOTAL CORPUS</div>
<div style="font-size:28px;font-weight:800;color:#00f5a0;letter-spacing:-0.5px">₹{total_corpus:,.0f}</div>
<div style="font-size:10px;color:rgba(255,255,255,0.25);margin-top:4px">spare change captured</div>
</div>
<div style="padding:20px;background:linear-gradient(145deg,rgba(0,212,255,0.08),rgba(0,212,255,0.02));border:1px solid rgba(0,212,255,0.15);border-radius:14px">
<div style="font-size:10px;color:rgba(255,255,255,0.35);font-family:DM Mono,monospace;letter-spacing:1.5px;margin-bottom:8px">MONTHLY AVG</div>
<div style="font-size:28px;font-weight:800;color:#00d4ff;letter-spacing:-0.5px">₹{monthly_avg:,.0f}</div>
<div style="font-size:10px;color:rgba(255,255,255,0.25);margin-top:4px">per month</div>
</div>
<div style="padding:20px;background:linear-gradient(145deg,rgba(245,158,11,0.08),rgba(245,158,11,0.02));border:1px solid rgba(245,158,11,0.15);border-radius:14px">
<div style="font-size:10px;color:rgba(255,255,255,0.35);font-family:DM Mono,monospace;letter-spacing:1.5px;margin-bottom:8px">TRANSACTIONS</div>
<div style="font-size:28px;font-weight:800;color:#f59e0b;letter-spacing:-0.5px">{total_txns:,}</div>
<div style="font-size:10px;color:rgba(255,255,255,0.25);margin-top:4px">rounded up</div>
</div>
<div style="padding:20px;background:linear-gradient(145deg,rgba(168,85,247,0.08),rgba(168,85,247,0.02));border:1px solid rgba(168,85,247,0.15);border-radius:14px">
<div style="font-size:10px;color:rgba(255,255,255,0.35);font-family:DM Mono,monospace;letter-spacing:1.5px;margin-bottom:8px">AVG / TXN</div>
<div style="font-size:28px;font-weight:800;color:#a855f7;letter-spacing:-0.5px">₹{avg_per_txn:.1f}</div>
<div style="font-size:10px;color:rgba(255,255,255,0.25);margin-top:4px">per transaction</div>
</div>
</div>''', unsafe_allow_html=True)
            st.markdown('<div style="margin:24px 40px 8px;font-family:DM Mono,monospace;font-size:10px;color:#00f5a0;letter-spacing:2px">MONTHLY ROUND-UP CORPUS</div>', unsafe_allow_html=True)
            import plotly.graph_objects as go
            fig_bar = go.Figure()
            fig_bar.add_trace(go.Bar(
                x=monthly_ru["DATE"].astype(str),
                y=monthly_ru["roundup_total"],
                marker_color="#00f5a0", opacity=0.85,
                hovertemplate="<b>%{x}</b><br>₹%{y:,.0f}<extra></extra>"
            ))
            fig_bar.update_layout(
                paper_bgcolor="#080d1a", plot_bgcolor="#0d1420",
                margin=dict(l=40,r=40,t=20,b=40), height=280,
                xaxis=dict(showgrid=False, color="#555"),
                yaxis=dict(gridcolor="#1a2035", color="#555",
                           tickprefix="₹", tickformat=",.0f"),
                showlegend=False
            )
            st.plotly_chart(fig_bar, use_container_width=True)
            st.markdown('<div style="margin:8px 40px;font-family:DM Mono,monospace;font-size:10px;color:#00f5a0;letter-spacing:2px">HISTORICAL XIRR SIMULATION (₹10 THRESHOLD)</div>', unsafe_allow_html=True)
            if MU_XIRR:
                xirr_cols = st.columns(len(MU_XIRR))
                INST_META = {"NIFTYBEES":("Nifty 50 ETF","#00f5a0"),"JUNIORBEES":("Nifty Next 50","#00d4ff"),"GOLDBEES":("Gold ETF","#f59e0b")}
                for col_ui, (inst_id, xdata) in zip(xirr_cols, MU_XIRR.items()):
                        name, color = INST_META.get(inst_id, (inst_id,"#fff"))
                        invested = xdata.get("total_invested",0)
                        current  = xdata.get("current_value",0)
                        xirr_v   = xdata.get("xirr_pct","—")
                        gain     = xdata.get("absolute_gain",0)
                        vs_fd    = xdata.get("vs_fd_delta",0)
                        col_ui.markdown(f'<div style="padding:16px;background:rgba(255,255,255,0.03);border:1px solid {color}30;border-radius:12px;margin-bottom:8px"><div style="font-size:11px;color:{color};font-family:DM Mono,monospace;margin-bottom:6px">{name}</div><div style="font-size:22px;font-weight:800;color:{color}">{xirr_v}%</div><div style="font-size:10px;color:rgba(255,255,255,0.4);margin-top:2px">XIRR</div><div style="margin-top:12px;font-size:12px;color:rgba(255,255,255,0.6)">Invested: ₹{invested:,.0f}</div><div style="font-size:12px;color:rgba(255,255,255,0.6)">Value: ₹{current:,.0f}</div><div style="font-size:12px;color:{"#00f5a0" if gain>0 else "#ff3c64"}">Gain: ₹{gain:+,.0f}</div><div style="font-size:11px;color:rgba(255,255,255,0.35);margin-top:4px">vs FD: ₹{vs_fd:+,.0f}</div></div>', unsafe_allow_html=True)
            st.markdown('<div style="margin:8px 40px;font-family:DM Mono,monospace;font-size:10px;color:#00f5a0;letter-spacing:2px">MONTE CARLO PROJECTION — 1000 SIMULATIONS</div>', unsafe_allow_html=True)
            thr_key = str(threshold)
            if MU_MC and thr_key in MU_MC:
                mc_data = MU_MC[thr_key]
                years_list = [5, 10, 15, 20]
                p25  = [mc_data["projections"][str(y)]["p25"]  for y in years_list]
                p50  = [mc_data["projections"][str(y)]["p50"]  for y in years_list]
                p75  = [mc_data["projections"][str(y)]["p75"]  for y in years_list]
                inv  = [mc_data["projections"][str(y)]["simple_invested"] for y in years_list]
                fig_mc = go.Figure()
                fig_mc.add_trace(go.Scatter(x=years_list+years_list[::-1],y=p75+p25[::-1],fill="toself",fillcolor="rgba(0,245,160,0.08)",line=dict(color="rgba(0,0,0,0)"),name="25th–75th pct",hoverinfo="skip"))
                fig_mc.add_trace(go.Scatter(x=years_list,y=p50,mode="lines+markers+text",line=dict(color="#00f5a0",width=3),marker=dict(size=8),text=[f'₹{v/100000:.0f}L' if v<10000000 else f'₹{v/10000000:.1f}Cr' for v in p50],textposition="top center",textfont=dict(color="#00f5a0",size=10),name="Median"))
                fig_mc.add_trace(go.Scatter(x=years_list,y=p75,mode="lines",line=dict(color="#00d4ff",width=1.5,dash="dash"),name="Optimistic (75th)"))
                fig_mc.add_trace(go.Scatter(x=years_list,y=p25,mode="lines",line=dict(color="#f59e0b",width=1.5,dash="dash"),name="Conservative (25th)"))
                fig_mc.add_trace(go.Scatter(x=years_list,y=inv,mode="lines",line=dict(color="rgba(255,255,255,0.3)",width=1.5,dash="dot"),name="Amount Invested"))
                fig_mc.update_layout(paper_bgcolor="#080d1a",plot_bgcolor="#0d1420",margin=dict(l=40,r=40,t=30,b=40),height=380,xaxis=dict(tickvals=years_list,ticktext=[f"{y}yr" for y in years_list],showgrid=False,color="#555"),yaxis=dict(gridcolor="#1a2035",color="#555",tickformat=",.0f",tickprefix="₹"),legend=dict(bgcolor="rgba(0,0,0,0)",font=dict(color="#aaa",size=10)),)
                st.plotly_chart(fig_mc, use_container_width=True)
            st.markdown('<div style="margin:24px 0 12px;font-family:DM Mono,monospace;font-size:10px;color:#00f5a0;letter-spacing:2px">WHERE TO INVEST YOUR ROUND-UPS</div>', unsafe_allow_html=True)
            if MU_INSTRUMENTS:
                for i in range(0, len(MU_INSTRUMENTS), 2):
                    row_insts = MU_INSTRUMENTS[i:i+2]
                    icols = st.columns(len(row_insts))
                    for ic, inst in zip(icols, row_insts):
                        r = inst.get("returns", {})
                        r1 = f'{r.get("1Y","—")}%' if r.get("1Y") else "—"
                        r3 = f'{r.get("3Y","—")}%' if r.get("3Y") else "—"
                        r5 = f'{r.get("5Y","—")}%' if r.get("5Y") else "—"
                        risk_color = inst.get("risk_color","#888")
                        tax_html = f'<span style="background:linear-gradient(135deg,#00f5a0,#00d4ff);color:#000;font-size:9px;font-weight:700;padding:3px 8px;border-radius:4px;margin-left:8px">80C</span>' if inst.get("tax_benefit") else ""
                        links = inst.get("platform_links", {})
                        link_btns = " ".join([f'<a href="{v}" target="_blank" style="display:inline-block;padding:5px 12px;background:rgba(255,255,255,0.06);border:1px solid rgba(255,255,255,0.1);border-radius:6px;color:#00d4ff;font-size:10px;font-family:DM Mono,monospace;text-decoration:none;letter-spacing:0.5px;transition:all 0.2s">{k.title()}</a>' for k,v in links.items()])
                        # Color the best return green
                        r_vals = [r.get("1Y",0) or 0, r.get("3Y",0) or 0, r.get("5Y",0) or 0]
                        r_colors = ["#e8eaf0","#e8eaf0","#e8eaf0"]
                        if max(r_vals) > 0:
                            r_colors[r_vals.index(max(r_vals))] = "#00f5a0"
                        ic.markdown(f'''<div style="padding:20px;background:linear-gradient(145deg,rgba(255,255,255,0.03),rgba(255,255,255,0.01));border:1px solid rgba(255,255,255,0.08);border-radius:16px;margin-bottom:14px;position:relative;overflow:hidden">
<div style="position:absolute;top:0;left:0;width:100%;height:3px;background:linear-gradient(90deg,{risk_color},{risk_color}80,transparent)"></div>
<div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:4px;margin-top:4px">
<div style="font-size:15px;font-weight:700;color:#e8eaf0">{inst["name"]}{tax_html}</div>
<div style="font-size:10px;color:{risk_color};font-family:DM Mono,monospace;background:{risk_color}15;padding:3px 8px;border-radius:4px">{inst["risk"]}</div>
</div>
<div style="font-size:11px;color:rgba(255,255,255,0.35);font-family:DM Mono,monospace;margin-bottom:12px">{inst["type"]} · {inst.get("fund_house","")}</div>
<div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:6px;margin-bottom:14px">
<div style="text-align:center;padding:8px 4px;background:rgba(255,255,255,0.04);border-radius:8px"><div style="font-size:9px;color:rgba(255,255,255,0.3);font-family:DM Mono,monospace;letter-spacing:1px">1Y</div><div style="font-size:16px;font-weight:700;color:{r_colors[0]};margin-top:2px">{r1}</div></div>
<div style="text-align:center;padding:8px 4px;background:rgba(255,255,255,0.04);border-radius:8px"><div style="font-size:9px;color:rgba(255,255,255,0.3);font-family:DM Mono,monospace;letter-spacing:1px">3Y</div><div style="font-size:16px;font-weight:700;color:{r_colors[1]};margin-top:2px">{r3}</div></div>
<div style="text-align:center;padding:8px 4px;background:rgba(255,255,255,0.04);border-radius:8px"><div style="font-size:9px;color:rgba(255,255,255,0.3);font-family:DM Mono,monospace;letter-spacing:1px">5Y</div><div style="font-size:16px;font-weight:700;color:{r_colors[2]};margin-top:2px">{r5}</div></div>
</div>
<div style="font-size:11px;color:rgba(255,255,255,0.45);line-height:1.5;margin-bottom:12px">{inst.get("description","")}</div>
<div style="display:flex;gap:8px;flex-wrap:wrap">{link_btns}</div>
</div>''', unsafe_allow_html=True)
            if MU_DISCLAIMER:
                st.markdown(f'<div style="margin:20px 0;padding:14px 18px;background:rgba(255,255,255,0.02);border-left:3px solid rgba(255,255,255,0.1);border-radius:0 8px 8px 0"><span style="font-size:10px;color:rgba(255,255,255,0.25);font-family:DM Mono,monospace;line-height:1.6">&#9878; SEBI DISCLAIMER · {MU_DISCLAIMER}</span></div>', unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown('<div style="border-top:1px solid rgba(255,255,255,0.06);margin-top:60px;padding:24px 40px;"><span style="font-family:DM Mono,monospace;font-size:10px;color:rgba(255,255,255,0.15);letter-spacing:2px;">FINSIGHT · CONTEXTUAL FLAGGING + TREND FORECAST + LLAMA 3.3 70B</span></div>', unsafe_allow_html=True)