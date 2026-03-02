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

# ── SmartCash: card databases ───────────────────────────────────────
import json as _json
import urllib.request as _urllib

@st.cache_data(ttl=3600)
def load_card_databases():
    _base = 'https://raw.githubusercontent.com/DarshMohapatra/FINSIGHT/main/'
    def _get(fname):
        try:
            with _urllib.urlopen(_base + fname, timeout=5) as resp:
                return _json.load(resp)
        except Exception:
            return []
    return _get('card_master.json'), _get('card_rewards.json')

SC_CARD_MASTER,  SC_CARD_REWARDS = load_card_databases()
SC_REWARDS_LOOKUP = {r['card_id']: r['rates'] for r in SC_CARD_REWARDS}
SC_CARD_LOOKUP    = {c['card_id']: c          for c in SC_CARD_MASTER}
SC_CARD_NAME      = {c['card_id']: c['bank'] + ' ' + c['card_name']
                     for c in SC_CARD_MASTER}
SC_CAT_MAP = {
    'Food & Dining':'Food & Dining', 'Grocery':'Grocery',
    'Shopping':'Shopping',           'Travel':'Travel',
    'Fuel':'Fuel',                   'Healthcare':'Healthcare',
    'Entertainment':'Entertainment', 'Utility':'Utility',
    'Salary':'Other',                'Other':'Other',
}


# ── SmartCash engine ─────────────────────────────────────────────────
def sc_get_best_card(amount, category, wallet):
    mapped = SC_CAT_MAP.get(category, 'Other')
    best_id, best_rate, best_cash = 'NONE', 0.0, 0.0
    for cid in wallet:
        rate = SC_REWARDS_LOOKUP.get(cid, {}).get(mapped,
               SC_REWARDS_LOOKUP.get(cid, {}).get('Other', 0))
        cash = round(amount * rate / 100, 2)
        if cash > best_cash:
            best_id, best_rate, best_cash = cid, rate, cash
    return {'best_card_id':best_id,
            'best_card_name': SC_CARD_NAME.get(best_id, best_id),
            'best_rate':best_rate,
            'best_cashback':best_cash,
            'baseline': round(amount * 1.0 / 100, 2)}

def sc_run_analysis(df, wallet):
    spend = df[(df['WITHDRAWAL AMT'] > 0) &
               df['CATEGORY'].notna() &
               (df['CATEGORY'] != 'Salary')].copy()
    rows = []
    for _, row in spend.iterrows():
        res = sc_get_best_card(row['WITHDRAWAL AMT'], row['CATEGORY'], wallet)
        rows.append({'DATE':row['DATE'],
                     'DESCRIPTION':row['TRANSACTION DETAILS'],
                     'CATEGORY':row['CATEGORY'],
                     'AMOUNT':row['WITHDRAWAL AMT'],
                     'BEST_CARD':res['best_card_name'],
                     'BEST_RATE':res['best_rate'],
                     'BEST_CASHBACK':res['best_cashback'],
                     'BASELINE':res['baseline'],
                     'EXTRA':round(res['best_cashback']-res['baseline'],2)})
    return pd.DataFrame(rows)

def sc_cat_summary(rdf):
    return (rdf.groupby('CATEGORY')
            .agg(spend=('AMOUNT','sum'), cashback=('BEST_CASHBACK','sum'),
                 baseline=('BASELINE','sum'), txns=('AMOUNT','count'),
                 best_card=('BEST_CARD', lambda x: x.mode()[0]),
                 avg_rate=('BEST_RATE','mean'))
            .assign(extra=lambda x: x['cashback']-x['baseline'])
            .sort_values('spend', ascending=False).reset_index())


    # ── SmartCash Tab ──────────────────────────────────────────────
    with tab5:
        st.markdown('<div style="padding:32px 40px 0"><div style="font-family:DM Mono,monospace;font-size:10px;color:#00f5a0;letter-spacing:3px;margin-bottom:8px">FEATURE 01 — SMARTCASH</div><div style="font-size:26px;font-weight:800;margin-bottom:8px">💳 Card Reward Maximiser</div><div style="color:rgba(255,255,255,0.4);font-size:14px;margin-bottom:24px">Find the best card in your wallet for every rupee you spend.</div></div>', unsafe_allow_html=True)
        if st.session_state.get('df') is None:
            st.info('Upload your bank statement in the UPLOAD tab first.')
        elif not SC_CARD_MASTER:
            st.error('Card database not loaded. Check GitHub for card_master.json')
        else:
            df_sc = st.session_state['df']
            st.markdown('<div style="margin:0 40px"><b>Step 1 — Select cards you own:</b></div>', unsafe_allow_html=True)
            card_opts = {c['bank']+' — '+c['card_name']+' ('+("FREE" if c['annual_fee']==0 else '₹'+str(c['annual_fee'])+'/yr')+')': c['card_id'] for c in SC_CARD_MASTER}
            sel = st.multiselect('Your credit cards:', list(card_opts.keys()), key='sc_wallet')
            wallet_sc = [card_opts[s] for s in sel]
            if st.button('⚡ Analyse Cashback Potential', key='sc_run', disabled=len(wallet_sc)==0):
                with st.spinner('Calculating best card for every transaction...'):
                    st.session_state['sc_results'] = sc_run_analysis(df_sc, wallet_sc)
                    st.session_state['sc_cat']     = sc_cat_summary(st.session_state['sc_results'])
            if st.session_state.get('sc_results') is not None:
                rdf  = st.session_state['sc_results']
                scat = st.session_state['sc_cat']
                tot_sp = rdf['AMOUNT'].sum()
                tot_cb = rdf['BEST_CASHBACK'].sum()
                tot_ex = rdf['EXTRA'].sum()
                eff_r  = (tot_cb/tot_sp*100) if tot_sp > 0 else 0
                c1,c2,c3,c4 = st.columns(4)
                c1.markdown(f'<div class="metric-card"><div class="metric-lbl">TOTAL SPEND</div><div class="metric-val">₹{tot_sp/1e5:.1f}L</div></div>', unsafe_allow_html=True)
                c2.markdown(f'<div class="metric-card"><div class="metric-lbl">BEST CASHBACK</div><div class="metric-val">₹{tot_cb:,.0f}</div></div>', unsafe_allow_html=True)
                c3.markdown(f'<div class="metric-card"><div class="metric-lbl">EXTRA vs 1% CARD</div><div class="metric-val">₹{tot_ex:,.0f}</div></div>', unsafe_allow_html=True)
                c4.markdown(f'<div class="metric-card"><div class="metric-lbl">EFFECTIVE RATE</div><div class="metric-val">{eff_r:.2f}%</div></div>', unsafe_allow_html=True)
                st.markdown('<div style="margin:24px 40px 8px"><b>Category Optimisation Guide:</b></div>', unsafe_allow_html=True)
                for _, row in scat.iterrows():
                    pct = min(int(row['avg_rate']/6*100),100)
                    st.markdown(f'<div style="margin:0 40px 8px;padding:14px 18px;background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.07);border-radius:12px"><div style="display:flex;justify-content:space-between;margin-bottom:6px"><b>{row["CATEGORY"]}</b><span style="color:#00f5a0;font-family:DM Mono,monospace;font-size:11px">Use → {row["best_card"]}</span></div><div style="color:rgba(255,255,255,0.4);font-size:12px;margin-bottom:8px">₹{row["spend"]:,.0f} spent · {row["txns"]:,} txns · earns ₹{row["cashback"]:,.0f} at {row["avg_rate"]:.1f}%</div><div style="background:rgba(255,255,255,0.06);border-radius:4px;height:4px"><div style="background:linear-gradient(90deg,#00f5a0,#00d4ff);width:{pct}%;height:4px;border-radius:4px"></div></div></div>', unsafe_allow_html=True)
                st.markdown('<div style="margin:20px 40px 8px"><b>Top 20 Transactions — Best Card to Use:</b></div>', unsafe_allow_html=True)
                top20 = rdf.nlargest(20,'BEST_CASHBACK')[['DATE','DESCRIPTION','CATEGORY','AMOUNT','BEST_CARD','BEST_RATE','BEST_CASHBACK']].copy()
                top20['AMOUNT']       = top20['AMOUNT'].apply(lambda x: f'₹{x:,.0f}')
                top20['BEST_CASHBACK']= top20['BEST_CASHBACK'].apply(lambda x: f'₹{x:,.0f}')
                top20['BEST_RATE']    = top20['BEST_RATE'].apply(lambda x: f'{x:.1f}%')
                st.dataframe(top20, use_container_width=True, hide_index=True)
