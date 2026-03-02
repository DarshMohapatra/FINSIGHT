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

# ── SmartCash: Load card databases from GitHub ────────────────────
import urllib.request as _urllib

def _load_json_from_github(filename):
    """Load a JSON file directly from the public GitHub repo."""
    try:
        base_url = "https://raw.githubusercontent.com/DarshMohapatra/FINSIGHT/main/"
        with _urllib.urlopen(base_url + filename, timeout=5) as resp:
            return json.load(resp)
    except Exception:
        return []

@st.cache_data(ttl=3600)   # cache for 1 hour — avoid re-fetching on every rerun
def load_card_databases():
    card_master  = _load_json_from_github("card_master.json")
    card_rewards = _load_json_from_github("card_rewards.json")
    return card_master, card_rewards

SC_CARD_MASTER, SC_CARD_REWARDS = load_card_databases()
SC_REWARDS_LOOKUP  = {r['card_id']: r['rates']    for r in SC_CARD_REWARDS}
SC_CARD_LOOKUP     = {c['card_id']: c             for c in SC_CARD_MASTER}
SC_CARD_NAME       = {c['card_id']: f"{c['bank']} {c['card_name']}"
                      for c in SC_CARD_MASTER}
SC_CATEGORY_MAP    = {
    "Food & Dining" : "Food & Dining",
    "Grocery"       : "Grocery",
    "Shopping"      : "Shopping",
    "Travel"        : "Travel",
    "Fuel"          : "Fuel",
    "Healthcare"    : "Healthcare",
    "Entertainment" : "Entertainment",
    "Utility"       : "Utility",
    "Salary"        : "Other",
    "Other"         : "Other",
}

    # ════════════════════════════════════════════════════════════
    # 💳 SMARTCASH TAB
    # ════════════════════════════════════════════════════════════
    with tab5:
        st.markdown("""
        <div style="padding:40px 40px 0;">
            <div style="font-family:DM Mono,monospace;font-size:10px;
                        color:#00f5a0;letter-spacing:3px;margin-bottom:8px;">
                FEATURE 01 — SMARTCASH
            </div>
            <div style="font-size:28px;font-weight:800;margin-bottom:8px;">
                💳 Card Reward Maximiser
            </div>
            <div style="color:rgba(255,255,255,0.4);font-size:14px;
                        margin-bottom:32px;">
                Find the best card in your wallet for every rupee you spend.
            </div>
        </div>
        """, unsafe_allow_html=True)

        if st.session_state.get('df') is None:
            st.markdown("""
            <div style="margin:40px;padding:32px;background:rgba(255,255,255,0.03);
                        border:1px solid rgba(255,255,255,0.08);border-radius:16px;
                        text-align:center;">
                <div style="font-size:40px;margin-bottom:16px;">📤</div>
                <div style="color:rgba(255,255,255,0.5);">
                    Upload your bank statement in the UPLOAD tab first
                </div>
            </div>
            """, unsafe_allow_html=True)

        elif not SC_CARD_MASTER:
            st.error("Card database not loaded. Check GitHub repo for card_master.json")

        else:
            df_sc = st.session_state['df']

            # ── Section 1: Wallet Setup ───────────────────────────
            st.markdown("""
            <div style="margin:0 40px 8px;">
                <div style="font-family:DM Mono,monospace;font-size:10px;
                            color:#00f5a0;letter-spacing:2px;margin-bottom:12px;">
                    STEP 01 — BUILD YOUR WALLET
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Build card options for multiselect
            card_options = {
                f"{c['bank']} — {c['card_name']} "
                f"({'FREE' if c['annual_fee']==0 else '₹'+str(c['annual_fee'])+'/yr'})": c['card_id']
                for c in SC_CARD_MASTER
            }

            with st.container():
                col_wallet, col_info = st.columns([2, 1])

                with col_wallet:
                    selected_labels = st.multiselect(
                        "Select credit cards you own:",
                        options   = list(card_options.keys()),
                        default   = [],
                        help      = "Only select cards you actually have — this determines your cashback analysis",
                        key       = "sc_wallet_select"
                    )
                    user_wallet_sc = [card_options[l] for l in selected_labels]

                with col_info:
                    if user_wallet_sc:
                        st.markdown(
                            f"""<div style="padding:16px;background:rgba(0,245,160,0.06);
                                border:1px solid rgba(0,245,160,0.15);border-radius:12px;
                                margin-top:28px;">
                                <div style="font-family:DM Mono,monospace;font-size:10px;
                                            color:#00f5a0;letter-spacing:2px;">
                                    WALLET READY
                                </div>
                                <div style="font-size:28px;font-weight:800;color:#00f5a0;">
                                    {len(user_wallet_sc)} cards
                                </div>
                            </div>""",
                            unsafe_allow_html=True
                        )

            # ── Run Analysis Button ───────────────────────────────
            st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
            run_sc = st.button("⚡ Analyse My Cashback Potential",
                                key="sc_run_btn",
                                disabled=len(user_wallet_sc) == 0)

            if run_sc and user_wallet_sc:
                with st.spinner("🔍 Calculating best card for every transaction..."):
                    sc_results  = sc_run_analysis(df_sc, user_wallet_sc)
                    sc_cat      = sc_category_summary(sc_results)
                    st.session_state['sc_results'] = sc_results
                    st.session_state['sc_cat']     = sc_cat

            # ── Section 2: Results ────────────────────────────────
            if st.session_state.get('sc_results') is not None:
                sc_results = st.session_state['sc_results']
                sc_cat     = st.session_state['sc_cat']

                total_spend    = sc_results['AMOUNT'].sum()
                total_cashback = sc_results['BEST_CASHBACK'].sum()
                total_extra    = sc_results['EXTRA'].sum()
                eff_rate       = (total_cashback/total_spend*100) if total_spend > 0 else 0

                # ── Headline metric cards ─────────────────────────
                st.markdown("<div style='margin:24px 40px 8px;'>", unsafe_allow_html=True)
                m1, m2, m3, m4 = st.columns(4)

                with m1:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-lbl">TOTAL SPEND</div>
                        <div class="metric-val">₹{total_spend/1e5:.1f}L</div>
                    </div>""", unsafe_allow_html=True)
                with m2:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-lbl">BEST CASHBACK</div>
                        <div class="metric-val">₹{total_cashback:,.0f}</div>
                    </div>""", unsafe_allow_html=True)
                with m3:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-lbl">EXTRA vs 1% CARD</div>
                        <div class="metric-val" style="-webkit-text-fill-color:#00f5a0">
                            ₹{total_extra:,.0f}
                        </div>
                    </div>""", unsafe_allow_html=True)
                with m4:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-lbl">EFFECTIVE RATE</div>
                        <div class="metric-val">{eff_rate:.2f}%</div>
                    </div>""", unsafe_allow_html=True)

                st.markdown("</div>", unsafe_allow_html=True)

                # ── Category optimisation guide ───────────────────
                st.markdown("""
                <div style="margin:24px 40px 8px;">
                    <div style="font-family:DM Mono,monospace;font-size:10px;
                                color:#00f5a0;letter-spacing:2px;margin-bottom:12px;">
                        STEP 02 — CATEGORY OPTIMISATION GUIDE
                    </div>
                </div>
                """, unsafe_allow_html=True)

                for _, row in sc_cat.iterrows():
                    pct_bar = min(int(row['avg_rate'] / 6 * 100), 100)
                    st.markdown(f"""
                    <div style="margin:0 40px 10px;padding:16px 20px;
                                background:rgba(255,255,255,0.03);
                                border:1px solid rgba(255,255,255,0.07);
                                border-radius:12px;">
                        <div style="display:flex;justify-content:space-between;
                                    align-items:center;margin-bottom:8px;">
                            <div style="font-weight:700;font-size:14px;">
                                {row['CATEGORY']}
                            </div>
                            <div style="font-family:DM Mono,monospace;font-size:11px;
                                        color:#00f5a0;">
                                Use → {row['best_card']}
                            </div>
                        </div>
                        <div style="display:flex;justify-content:space-between;
                                    color:rgba(255,255,255,0.4);font-size:12px;
                                    margin-bottom:8px;">
                            <span>₹{row['spend']:,.0f} spent · {row['txns']:,} txns</span>
                            <span>Earns ₹{row['cashback']:,.0f} · {row['avg_rate']:.1f}% avg</span>
                        </div>
                        <div style="background:rgba(255,255,255,0.06);
                                    border-radius:4px;height:4px;">
                            <div style="background:linear-gradient(90deg,#00f5a0,#00d4ff);
                                        width:{pct_bar}%;height:4px;border-radius:4px;">
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                # ── Top transactions table ────────────────────────
                st.markdown("""
                <div style="margin:24px 40px 8px;">
                    <div style="font-family:DM Mono,monospace;font-size:10px;
                                color:#00f5a0;letter-spacing:2px;margin-bottom:12px;">
                        STEP 03 — TOP 20 TRANSACTIONS
                    </div>
                </div>
                """, unsafe_allow_html=True)

                top20 = (sc_results
                         .nlargest(20, 'BEST_CASHBACK')
                         [['DATE','DESCRIPTION','CATEGORY',
                           'AMOUNT','BEST_CARD','BEST_RATE','BEST_CASHBACK']]
                         .rename(columns={
                             'BEST_CARD'    : 'Use This Card',
                             'BEST_RATE'    : 'Rate %',
                             'BEST_CASHBACK': 'Earns ₹'
                         })
                        )
                top20['AMOUNT']  = top20['AMOUNT'].apply(lambda x: f"₹{x:,.0f}")
                top20['Earns ₹'] = top20['Earns ₹'].apply(lambda x: f"₹{x:,.0f}")
                top20['Rate %']  = top20['Rate %'].apply(lambda x: f"{x:.1f}%")

                st.dataframe(
                    top20,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "DESCRIPTION": st.column_config.TextColumn(width="large"),
                        "Use This Card": st.column_config.TextColumn(width="medium"),
                    }
                )

