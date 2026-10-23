"""
📈 Stock Opportunity Scanner  (v2 - scoring rééquilibré)
---------------------------------------------------------
Application Streamlit qui scanne une watchlist d'actions premium + les positions
du portefeuille, calcule un score d'opportunité d'achat et envoie un email
automatique quand une action change de catégorie ou voit son score progresser.

SCORING V2 (total direct sur 100, plus de renormalisation 135 -> 100) :
    Timing (80 pts)
      - RSI           25 : 15 niveau (survente) + 10 retournement (rebond depuis la survente)
      - MACD          20 : 10 niveau de l'histogramme + 10 accélération (pente sur 3 jours)
      - Volume        10 : ratio 3 derniers jours / moyenne 20j (divisé par 2 si prix en baisse)
      - Drawdown      15 : baisse depuis le plus haut 6 mois
      - Support SMA50 10 : zone idéale = proche de la SMA50 (-3% / +3%), pas trop étendu
    Qualité (20 pts)
      - Croissance CA  8 | Marge brute 5 | Marge nette 7
      - Donnée indisponible (ETF, titres étrangers...) => score neutre (50 %), pas de pénalité

Installation :
    pip install streamlit yfinance pandas numpy streamlit-autorefresh supabase requests

Lancement :
    streamlit run stock_scanner.py

Configuration : secrets Streamlit SUPABASE_URL, SUPABASE_KEY, RESEND_API_KEY,
RESEND_FROM_EMAIL, GROQ_API_KEY (chatbot), GROQ_MODEL (optionnel).
"""

from datetime import datetime, date

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
import os
import re
import requests
from supabase import create_client

try:
    from streamlit_autorefresh import st_autorefresh
    AUTOREFRESH_AVAILABLE = True
except ImportError:
    AUTOREFRESH_AVAILABLE = False

st.set_page_config(page_title="Stock Opportunity Scanner", page_icon="📈", layout="wide", initial_sidebar_state="expanded")

# ============================================================
# CONFIGURATION PERSISTANTE SUPABASE + EMAIL RESEND
# ============================================================
def get_setting(name, default=""):
    try:
        return st.secrets.get(name, os.getenv(name, default))
    except Exception:
        return os.getenv(name, default)

SUPABASE_URL = get_setting("SUPABASE_URL")
SUPABASE_KEY = get_setting("SUPABASE_KEY")
RESEND_API_KEY = get_setting("RESEND_API_KEY")
RESEND_FROM_EMAIL = get_setting("RESEND_FROM_EMAIL")
GROQ_API_KEY = get_setting("GROQ_API_KEY")
GROQ_MODEL = get_setting("GROQ_MODEL", "openai/gpt-oss-20b") or "openai/gpt-oss-20b"

@st.cache_resource
def get_supabase_client():
    if not SUPABASE_URL or not SUPABASE_KEY:
        return None
    try:
        return create_client(SUPABASE_URL, SUPABASE_KEY)
    except Exception:
        return None

supabase = get_supabase_client()

def load_recipient_email():
    if supabase is None:
        return ""
    try:
        rows = supabase.table("app_settings").select("recipient_email").eq("id", 1).limit(1).execute().data
        return rows[0].get("recipient_email", "") if rows else ""
    except Exception:
        return ""

def save_recipient_email(email):
    if supabase is None:
        return False, "Supabase n'est pas configuré. Vérifie SUPABASE_URL et SUPABASE_KEY."
    try:
        supabase.table("app_settings").upsert({"id": 1, "recipient_email": email}).execute()
        return True, None
    except Exception as exc:
        return False, str(exc)

def load_persistent_alert_state():
    if supabase is None:
        return {}
    try:
        rows = supabase.table("ticker_alert_state").select("ticker,score,bucket,last_alert_time").execute().data
        state = {}
        for row in rows:
            last = row.get("last_alert_time")
            if last:
                try:
                    last = datetime.fromisoformat(last.replace("Z", "+00:00"))
                    if last.tzinfo is not None:
                        last = last.replace(tzinfo=None)
                except Exception:
                    last = None
            state[row["ticker"]] = {"score": float(row["score"]), "bucket": row["bucket"], "last_alert_time": last}
        return state
    except Exception:
        return {}

def persist_alert_state(ticker, state):
    if supabase is None:
        return
    try:
        last = state.get("last_alert_time")
        supabase.table("ticker_alert_state").upsert({
            "ticker": ticker, "score": float(state["score"]), "bucket": state["bucket"],
            "last_alert_time": last.isoformat() if last else None,
            "updated_at": datetime.utcnow().isoformat() + "Z"
        }).execute()
    except Exception as exc:
        st.warning(f"Impossible de sauvegarder l'état de {ticker} dans Supabase : {exc}")

st.markdown("""
<style>
    :root { color-scheme: light; }
    .stApp {
        background: radial-gradient(ellipse at 8% 0%, rgba(99,102,241,.075), transparent 34%),
                    radial-gradient(ellipse at 95% 8%, rgba(14,165,233,.055), transparent 30%), #f7f9fc;
        color: #182338;
    }
    .block-container { padding-top: 1.6rem; padding-bottom: 3.5rem; max-width: 1480px; }
    [data-testid="stHeader"] { background: rgba(247,249,252,.88); backdrop-filter: blur(12px); }
    [data-testid="stSidebar"] { background: rgba(255,255,255,.94); border-right: 1px solid #e7ebf3; }
    [data-testid="stSidebar"] * { color: #26344b; }
    h1 { letter-spacing: -.045em; font-weight: 820 !important; color: #111827; }
    h2, h3 { letter-spacing: -.03em; color: #172033; font-weight: 730 !important; }
    p, label, [data-testid="stMarkdownContainer"] { color: #344258; }
    [data-testid="stCaptionContainer"] { color: #748198; }
    [data-testid="stMetric"] {
        background: linear-gradient(145deg, #fff, #f9fbff); border: 1px solid #e5eaf3;
        padding: 16px 17px; border-radius: 18px; box-shadow: 0 5px 18px rgba(24,39,75,.035);
        transition: transform .18s ease, box-shadow .18s ease, border-color .18s ease;
    }
    [data-testid="stMetric"]:hover { transform: translateY(-2px); border-color: #d3def0; box-shadow: 0 10px 24px rgba(24,39,75,.065); }
    [data-testid="stMetricLabel"] { color: #728097 !important; }
    [data-testid="stMetricValue"] { color: #111827 !important; font-weight: 780; letter-spacing: -.04em; }
    [data-testid="stTabs"] [role="tablist"] { gap: 7px; border-bottom: 1px solid #e5eaf3; }
    [data-testid="stTabs"] button { border-radius: 11px 11px 0 0; padding: .75rem 1rem; transition: background .18s ease, color .18s ease; }
    [data-testid="stTabs"] button[aria-selected="true"] { background: rgba(59,130,246,.075); color: #1d4ed8; }
    [data-testid="stExpander"] { border: 1px solid #e5eaf3; border-radius: 16px; background: rgba(255,255,255,.9); overflow: hidden; }
    [data-testid="stVerticalBlockBorderWrapper"] { border-color: #e5eaf3 !important; border-radius: 18px !important; background: rgba(255,255,255,.9); }
    div.stButton > button, [data-testid="stFormSubmitButton"] button {
        border-radius: 12px; border: 1px solid #dce4f0; transition: all .18s ease;
        min-height: 2.6rem; font-weight: 650; box-shadow: 0 2px 5px rgba(24,39,75,.025);
    }
    div.stButton > button:hover, [data-testid="stFormSubmitButton"] button:hover {
        border-color: #9bb9ee; color: #1d4ed8; transform: translateY(-1px); box-shadow: 0 5px 13px rgba(37,99,235,.09);
    }
    [data-testid="stTextInput"] input, [data-testid="stNumberInput"] input,
    [data-testid="stSelectbox"] [role="combobox"], [data-testid="stMultiSelect"] [role="combobox"] { border-radius: 12px; }
    [data-testid="stChatMessage"] {
        border: 1px solid #e7ecf4; border-radius: 18px; padding: 1rem 1.1rem; margin: .65rem 0;
        background: rgba(255,255,255,.9); box-shadow: 0 4px 14px rgba(24,39,75,.025);
    }
    [data-testid="stChatInput"] { border-radius: 16px; }
    .score-badge { display:inline-flex; align-items:center; justify-content:center; padding:5px 11px; border-radius:999px; font-weight:750; font-size:.82rem; white-space:nowrap; }
    .badge-green { background:#dcfce7; color:#166534; border:1px solid #bbf7d0; }
    .badge-yellow { background:#fef3c7; color:#92400e; border:1px solid #fde68a; }
    .badge-red { background:#fee2e2; color:#991b1b; border:1px solid #fecaca; }
    .stock-card {
        background: linear-gradient(135deg, #fff, #f9fbff); border:1px solid #e5eaf3; border-radius:18px;
        padding:19px 20px 14px; margin:10px 0 8px; box-shadow:0 5px 18px rgba(24,39,75,.035);
        transition: transform .18s ease, border-color .18s ease, box-shadow .18s ease;
    }
    .stock-card:hover { transform:translateY(-2px); border-color:#c7d7f0; box-shadow:0 12px 28px rgba(24,39,75,.075); }
    .stock-card-top { display:flex; justify-content:space-between; align-items:flex-start; gap:12px; margin-bottom:14px; }
    .stock-ticker { font-size:1.08rem; font-weight:800; color:#172033; letter-spacing:-.025em; }
    .stock-company { font-size:.84rem; color:#748198; margin-top:3px; }
    .stock-meta { display:flex; flex-wrap:wrap; gap:7px; margin-top:8px; }
    .stock-chip { background:#f3f6fb; color:#4b5b73; border:1px solid #e6ebf3; border-radius:9px; padding:4px 9px; font-size:.75rem; }
    .stock-bars { display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:16px; }
    .stock-bar-label { display:flex; justify-content:space-between; gap:8px; color:#748198; font-size:.76rem; margin-bottom:7px; }
    .stock-track { height:7px; border-radius:99px; background:#e9eef6; overflow:hidden; }
    .stock-fill { height:100%; border-radius:99px; }
    .stock-foot { display:flex; flex-wrap:wrap; gap:16px; color:#748198; font-size:.78rem; margin-top:14px; padding-top:11px; border-top:1px solid #edf1f7; }
    .detail-row { display:grid; grid-template-columns:minmax(140px,1.2fr) minmax(100px,2fr) 65px; align-items:center; gap:12px; padding:10px 0; border-bottom:1px solid #edf1f7; }
    .detail-label { color:#475569; font-size:.86rem; }
    .detail-track { height:8px; background:#e9eef6; border-radius:99px; overflow:hidden; }
    .detail-value { text-align:right; color:#172033; font-size:.83rem; font-variant-numeric:tabular-nums; }
    .section-kicker { color:#748198; text-transform:uppercase; letter-spacing:.12em; font-size:.7rem; font-weight:750; }
    [data-testid="stLineChart"], [data-testid="stPlotlyChart"] { border:1px solid #e5eaf3; border-radius:18px; padding:7px; background:rgba(255,255,255,.9); }
    hr { border-color:#e6ebf3; }
    @media (max-width:700px) { .block-container {padding-top:1rem;} .stock-bars {grid-template-columns:1fr;gap:10px;} .stock-card {padding:15px;} .detail-row {grid-template-columns:minmax(105px,1fr) minmax(70px,1fr) 55px;gap:8px;} }
</style>
""", unsafe_allow_html=True)


# ============================================================
# LANGUE DE L'INTERFACE
# ============================================================
with st.sidebar:
    app_language = st.selectbox("🌐 Language / Langue", ["Français", "English"], key="app_language")
IS_EN = app_language == "English"
def tr(fr, en):
    return en if IS_EN else fr

st.title("📈 Stock Opportunity Scanner")
st.caption(tr("Détection automatique des meilleurs moments d'achat sur ta watchlist premium", "Automatically detect potential buying opportunities in your premium watchlist"))

# ============================================================
# WATCHLIST
# ============================================================
WATCHLIST = {
    "NVDA": ("NVIDIA", "Semi-conducteurs"),
    "AMD": ("Advanced Micro Devices", "Semi-conducteurs"),
    "AVGO": ("Broadcom", "Semi-conducteurs"),
    "TSM": ("Taiwan Semiconductor", "Semi-conducteurs"),
    "MU": ("Micron Technology", "Semi-conducteurs"),
    "MRVL": ("Marvell Technology", "Semi-conducteurs"),
    "QCOM": ("Qualcomm", "Semi-conducteurs"),
    "ARM": ("Arm Holdings", "Semi-conducteurs"),
    "ASML": ("ASML", "Semi-conducteurs"),
    "AMAT": ("Applied Materials", "Semi-conducteurs"),
    "LRCX": ("Lam Research", "Semi-conducteurs"),
    "KLAC": ("KLA", "Semi-conducteurs"),
    "ORCL": ("Oracle", "Software/Cloud"),
    "PLTR": ("Palantir Technologies", "Software/Cloud"),
    "SNOW": ("Snowflake", "Software/Cloud"),
    "NET": ("Cloudflare", "Software/Cloud"),
    "DDOG": ("Datadog", "Software/Cloud"),
    "MDB": ("MongoDB", "Software/Cloud"),
    "NOW": ("ServiceNow", "Software/Cloud"),
    "CRM": ("Salesforce", "Software/Cloud"),
    "CRWD": ("CrowdStrike", "Cybersécurité"),
    "PANW": ("Palo Alto Networks", "Cybersécurité"),
    "ZS": ("Zscaler", "Cybersécurité"),
    "FTNT": ("Fortinet", "Cybersécurité"),
    "GOOGL": ("Alphabet", "Big Tech"),
    "AMZN": ("Amazon", "Big Tech"),
    "META": ("Meta Platforms", "Big Tech"),
    "AAPL": ("Apple", "Big Tech"),
    "MSFT": ("Microsoft", "Big Tech"),
    "TSLA": ("Tesla", "Big Tech"),
    "RKLB": ("Rocket Lab USA", "Spatial/Défense"),
    "ASTS": ("AST SpaceMobile", "Spatial/Défense"),
    "KTOS": ("Kratos Defense", "Spatial/Défense"),
    "LMT": ("Lockheed Martin", "Spatial/Défense"),
    "RTX": ("RTX", "Spatial/Défense"),
    "UBER": ("Uber Technologies", "Consommation/Fintech"),
    "ABNB": ("Airbnb", "Consommation/Fintech"),
    "MELI": ("MercadoLibre", "Consommation/Fintech"),
    "V": ("Visa", "Consommation/Fintech"),
    "MA": ("Mastercard", "Consommation/Fintech"),
    "ADBE": ("Adobe", "Logiciel/IA"),
    "INTU": ("Intuit", "Fintech/SaaS"),
    "SHOP": ("Shopify", "E-commerce"),
    "ENPH": ("Enphase Energy", "Énergie solaire"),
    "FSLR": ("First Solar", "Énergie propre"),
    "CELH": ("Celsius Holdings", "Consommation"),
    "LULU": ("Lululemon Athletica", "Consommation"),
    "NKE": ("Nike", "Consommation"),
    "DIS": ("Walt Disney Company", "Média"),
    "PYPL": ("PayPal", "Fintech"),
    "SQ": ("Block", "Fintech"),
    "CRSP": ("CRISPR Therapeutics", "Biotech"),
    "ISRG": ("Intuitive Surgical", "Santé"),
    "TMO": ("Thermo Fisher Scientific", "Santé"),
    "DE": ("Deere & Company", "Industrie"),
    # Actions spéculatives : forte volatilité / risque élevé
    "IONQ": ("IonQ", "Spéculatif - Informatique quantique"),
    "RGTI": ("Rigetti Computing", "Spéculatif - Informatique quantique"),
    "QBTS": ("D-Wave Quantum", "Spéculatif - Informatique quantique"),
    "QUBT": ("Quantum Computing Inc.", "Spéculatif - Informatique quantique"),
    "LUNR": ("Intuitive Machines", "Spéculatif - Spatial"),
    "ACHR": ("Archer Aviation", "Spéculatif - eVTOL"),
    "JOBY": ("Joby Aviation", "Spéculatif - eVTOL"),
    "OKLO": ("Oklo", "Spéculatif - Nucléaire"),
    "SMR": ("NuScale Power", "Spéculatif - Nucléaire"),
    "SERV": ("Serve Robotics", "Spéculatif - Robotique"),
    "TEM": ("Tempus AI", "Spéculatif - IA/Santé"),
    "RXRX": ("Recursion Pharmaceuticals", "Spéculatif - Biotech/IA"),
    "HIMS": ("Hims & Hers Health", "Croissance spéculative - Santé"),
    "SOUN": ("SoundHound AI", "Spéculatif - IA"),
    "BBAI": ("BigBear.ai", "Spéculatif - IA/Défense"),
    "AI": ("C3.ai", "Spéculatif - IA"),
    "DNA": ("Ginkgo Bioworks", "Spéculatif - Biotechnologie"),
    "ENVX": ("Enovix", "Spéculatif - Batteries"),
    "PL": ("Planet Labs", "Spéculatif - Spatial/Données"),
    "RDW": ("Redwire", "Spéculatif - Spatial/Défense"),
}

PORTFOLIO_EXTRA = {
    "NVNO": ("enVVeno Medical", "Portefeuille - Spéculatif"),
    "OTLK": ("Outlook Therapeutics", "Portefeuille - Spéculatif"),
    "BNP.PA": ("BNP Paribas", "Portefeuille - Equity"),
}

ALL_ASSETS = {**WATCHLIST, **PORTFOLIO_EXTRA}

# ============================================================
# BARÈME (points max par critère) - total = 100
# ============================================================
MAX_RSI_LEVEL, MAX_RSI_TURN = 15, 10
MAX_RSI = MAX_RSI_LEVEL + MAX_RSI_TURN          # 25
MAX_MACD_LEVEL, MAX_MACD_SLOPE = 10, 10
MAX_MACD = MAX_MACD_LEVEL + MAX_MACD_SLOPE      # 20
MAX_VOL = 10
MAX_DRAWDOWN = 15
MAX_SUPPORT = 10
MAX_GROWTH, MAX_MARGIN, MAX_PROFIT = 8, 5, 7    # 20 au total

# ============================================================
# INDICATEURS TECHNIQUES
# ============================================================
def compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """RSI de Wilder (lissage exponentiel alpha=1/period), plus fidèle que la moyenne simple."""
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)


def compute_macd(close: pd.Series, fast=12, slow=26, signal=9):
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line, signal_line


def scaled_score(value: float, worst: float, best: float, max_points: float) -> float:
    """
    Interpolation linéaire entre deux bornes (fonctionne dans les deux sens :
    best > worst = "plus haut, mieux" ; best < worst = "plus bas, mieux", ex: RSI).
    """
    if best == worst:
        return max_points if value >= best else 0.0
    frac = (value - worst) / (best - worst)
    frac = max(0.0, min(1.0, frac))
    return round(frac * max_points, 2)


def band_score(value: float, zero_low: float, full_low: float, full_high: float,
               zero_high: float, max_points: float) -> float:
    """Score en trapèze : 0 hors de [zero_low, zero_high], plein dans [full_low, full_high]."""
    if value <= zero_low or value >= zero_high:
        return 0.0
    if full_low <= value <= full_high:
        return float(max_points)
    if value < full_low:
        return scaled_score(value, zero_low, full_low, max_points)
    return scaled_score(value, zero_high, full_high, max_points)


@st.cache_data(ttl=900, show_spinner=False)
def fetch_history(ticker: str) -> pd.DataFrame:
    try:
        df = yf.download(ticker, period="9mo", interval="1d", progress=False, auto_adjust=True)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=86400, show_spinner=False)
def fetch_fundamentals(ticker: str) -> dict:
    """Fractions (0.15 = 15 %) ou None si indisponible. Cache 24h."""
    try:
        info = yf.Ticker(ticker).info
        return {
            "revenue_growth": info.get("revenueGrowth"),
            "gross_margin": info.get("grossMargins"),
            "profit_margin": info.get("profitMargins"),
        }
    except Exception:
        return {"revenue_growth": None, "gross_margin": None, "profit_margin": None}


def fundamental_points(value, worst, best, max_points):
    """Points continus ; si la donnée est absente -> score neutre (50 %), sans pénalité."""
    if value is None:
        return round(max_points / 2, 2)
    return scaled_score(value * 100, worst, best, max_points)


def analyze_ticker(ticker: str) -> dict | None:
    df = fetch_history(ticker)
    if df.empty or len(df) < 60 or "Close" not in df:
        return None

    close = df["Close"].dropna()
    volume = df["Volume"].dropna()
    last_price = float(close.iloc[-1])

    # --- RSI : niveau (survente) + retournement (rebond depuis la survente) ---
    rsi_series = compute_rsi(close)
    rsi = float(rsi_series.iloc[-1])
    rsi_delta3 = float(rsi - rsi_series.iloc[-4])
    rsi_min10 = float(rsi_series.iloc[-10:].min())
    # Niveau : RSI <= 25 -> 15 pts ; RSI >= 50 -> 0
    rsi_level_score = scaled_score(rsi, worst=50, best=25, max_points=MAX_RSI_LEVEL)
    # Retournement : compte seulement si le RSI est passé sous 40 ces 10 derniers jours,
    # et croît de 0 (0 pt) à +8 points en 3 jours (10 pts)
    rsi_turn_score = (
        scaled_score(rsi_delta3, worst=0, best=8, max_points=MAX_RSI_TURN)
        if rsi_min10 < 40 else 0.0
    )
    rsi_score = round(rsi_level_score + rsi_turn_score, 2)

    # --- MACD : niveau de l'histogramme + accélération sur 3 jours ---
    macd_line, signal_line = compute_macd(close)
    hist = macd_line - signal_line
    macd_hist = float(hist.iloc[-1])
    macd_hist_pct = (macd_hist / last_price) * 100 if last_price else 0.0
    macd_slope_pct = ((macd_hist - float(hist.iloc[-4])) / last_price) * 100 if last_price else 0.0
    macd_bullish = macd_hist > 0
    # Niveau : -0.5 % -> 0 ; +0.5 % -> 10 pts
    macd_level_score = scaled_score(macd_hist_pct, worst=-0.5, best=0.5, max_points=MAX_MACD_LEVEL)
    # Pente : histogramme qui monte de -0.1 % (0 pt) à +0.4 % (10 pts) en 3 jours
    macd_slope_score = scaled_score(macd_slope_pct, worst=-0.1, best=0.4, max_points=MAX_MACD_SLOPE)
    macd_score = round(macd_level_score + macd_slope_score, 2)

    # --- Volume : 3 derniers jours vs moyenne 20j précédente (évite le biais de la séance en cours) ---
    vol3 = volume.iloc[-3:].mean()
    vol_avg20 = volume.iloc[-23:-3].mean()
    vol_ratio = float(vol3 / vol_avg20) if pd.notna(vol_avg20) and vol_avg20 > 0 else 0.0
    vol_above_avg = vol_ratio > 1.0
    ret3 = float(close.iloc[-1] / close.iloc[-4] - 1)
    # 0.8x -> 0 ; 1.8x -> 10 pts. Volume sur un titre qui baisse = distribution : points divisés par 2
    vol_score = scaled_score(vol_ratio, worst=0.8, best=1.8, max_points=MAX_VOL)
    if ret3 < 0:
        vol_score = round(vol_score * 0.5, 2)

    # --- Drawdown (plus haut ~6 mois) ---
    recent_high = close.rolling(126, min_periods=30).max().iloc[-1]
    drawdown_pct = float((recent_high - last_price) / recent_high * 100) if recent_high else 0.0
    drawdown_ok = drawdown_pct > 15
    # 5 % -> 0 ; 30 % et plus -> 15 pts
    drawdown_score = scaled_score(drawdown_pct, worst=5, best=30, max_points=MAX_DRAWDOWN)

    # --- Support : zone idéale = proche de la SMA50 (en %) ---
    sma50 = close.rolling(50, min_periods=30).mean().iloc[-1]
    pct_vs_sma50 = float((last_price - sma50) / sma50 * 100) if pd.notna(sma50) and sma50 > 0 else -100.0
    above_support = pct_vs_sma50 > 0
    # Plein entre -3 % et +3 % ; dégressif jusqu'à 0 pt à -12 % (couteau qui tombe) et +15 % (titre étendu)
    support_score = band_score(pct_vs_sma50, zero_low=-12, full_low=-3, full_high=3,
                               zero_high=15, max_points=MAX_SUPPORT)

    # --- Fondamentaux (filtre qualité, 20 pts) ---
    fundamentals = fetch_fundamentals(ticker)
    revenue_growth = fundamentals["revenue_growth"]
    gross_margin = fundamentals["gross_margin"]
    profit_margin = fundamentals["profit_margin"]

    # Bornes relevées pour ne plus saturer dès que l'entreprise est "bonne"
    # Croissance CA : 0 % -> 0 ; 40 % -> 8 pts
    growth_score = fundamental_points(revenue_growth, worst=0, best=40, max_points=MAX_GROWTH)
    # Marge brute : 20 % -> 0 ; 70 % -> 5 pts
    margin_score = fundamental_points(gross_margin, worst=20, best=70, max_points=MAX_MARGIN)
    # Marge nette : -10 % -> 0 ; 25 % -> 7 pts
    profit_score = fundamental_points(profit_margin, worst=-10, best=25, max_points=MAX_PROFIT)

    raw_total = (
        rsi_score + macd_score + vol_score + drawdown_score + support_score
        + growth_score + margin_score + profit_score
    )
    score = round(raw_total, 1)  # déjà sur 100

    return {
        "ticker": ticker,
        "price": last_price,
        "rsi": rsi,
        "rsi_delta3": rsi_delta3,
        "rsi_level_score": rsi_level_score,
        "rsi_turn_score": rsi_turn_score,
        "rsi_score": rsi_score,
        "macd_bullish": macd_bullish,
        "macd_hist_pct": macd_hist_pct,
        "macd_slope_pct": macd_slope_pct,
        "macd_score": macd_score,
        "vol_above_avg": vol_above_avg,
        "vol_ratio": vol_ratio,
        "vol_score": vol_score,
        "drawdown_pct": drawdown_pct,
        "drawdown_ok": drawdown_ok,
        "drawdown_score": drawdown_score,
        "above_support": above_support,
        "pct_vs_sma50": pct_vs_sma50,
        "support_score": support_score,
        "revenue_growth": revenue_growth,
        "gross_margin": gross_margin,
        "profit_margin": profit_margin,
        "growth_score": growth_score,
        "margin_score": margin_score,
        "profit_score": profit_score,
        "raw_total": round(raw_total, 1),
        "score": score,
    }


# ============================================================
# EMAIL (mise en forme HTML)
# ============================================================
BUCKET_LABELS = {
    "green": ("🟢", "Achat potentiel", "#22c55e"),
    "yellow": ("🟡", "À surveiller", "#eab308"),
    "red": ("🔴", "Rien à faire", "#ef4444"),
}
BUCKET_TEXT = ({
    "green": "🟢 Potential buy", "yellow": "🟡 Watchlist", "red": "🔴 No action",
} if IS_EN else {k: f"{v[0]} {v[1]}" for k, v in BUCKET_LABELS.items()})


def _build_email_html(items: list[dict]) -> str:
    rows_html = []
    for item in items:
        r = item["result"]
        emoji, label, color = BUCKET_LABELS[item["bucket"]]
        prev_score_txt = f"{item['prev_score']:.1f}" if item["prev_score"] is not None else "—"
        rows_html.append(f"""
        <tr>
          <td style="padding:16px; border:1px solid #2a2f3a; border-radius:10px; display:block; margin-bottom:12px; background:#161a23;">
            <table width="100%" cellpadding="0" cellspacing="0">
              <tr>
                <td style="font-size:16px; font-weight:700; color:#f5f5f5;">
                  {r['ticker']} <span style="color:#9aa0ab; font-weight:400; font-size:13px;">— {r['name']}</span>
                </td>
                <td align="right">
                  <span style="display:inline-block; padding:4px 12px; border-radius:999px;
                        background:{color}22; color:{color}; font-weight:700; font-size:13px;">
                    {emoji} {label}
                  </span>
                </td>
              </tr>
              <tr>
                <td colspan="2" style="padding-top:8px; font-size:13px; color:#c9ced8;">
                  {item['reason']}
                </td>
              </tr>
              <tr>
                <td colspan="2" style="padding-top:10px; font-size:13px; color:#9aa0ab;">
                  Score : <b style="color:#f5f5f5;">{r['score']:.1f}/100</b>
                  <span style="color:#6b7280;"> (précédent : {prev_score_txt})</span>
                  &nbsp;·&nbsp; Prix : <b style="color:#f5f5f5;">{r['price']:.2f}</b>
                  &nbsp;·&nbsp; RSI : <b style="color:#f5f5f5;">{r['rsi']:.0f}</b>
                  &nbsp;·&nbsp; Drawdown : <b style="color:#f5f5f5;">{r['drawdown_pct']:.1f}%</b>
                </td>
              </tr>
            </table>
          </td>
        </tr>
        """)

    return f"""
    <html>
      <body style="margin:0; padding:0; background:#0e1117; font-family:Segoe UI, Arial, sans-serif;">
        <table width="100%" cellpadding="0" cellspacing="0" style="background:#0e1117; padding:24px 0;">
          <tr>
            <td align="center">
              <table width="600" cellpadding="0" cellspacing="0">
                <tr>
                  <td style="padding-bottom:18px;">
                    <div style="font-size:20px; font-weight:800; color:#f5f5f5;">📈 Stock Opportunity Scanner</div>
                    <div style="font-size:13px; color:#9aa0ab; margin-top:4px;">
                      {len(items)} mouvement(s) détecté(s) le {date.today().strftime('%d/%m/%Y')}
                    </div>
                  </td>
                </tr>
                {''.join(rows_html)}
                <tr>
                  <td style="padding-top:12px; font-size:11px; color:#6b7280;">
                    Alerte générée automatiquement par ton Stock Opportunity Scanner.
                  </td>
                </tr>
              </table>
            </td>
          </tr>
        </table>
      </body>
    </html>
    """


def _build_email_text(items: list[dict]) -> str:
    lines = [f"{len(items)} mouvement(s) détecté(s) le {date.today().strftime('%d/%m/%Y')}", ""]
    for item in items:
        r = item["result"]
        _, label, _ = BUCKET_LABELS[item["bucket"]]
        prev_score_txt = f"{item['prev_score']:.1f}" if item["prev_score"] is not None else "—"
        lines.append(
            f"- {r['ticker']} ({r['name']}) : {label} | score {r['score']:.1f}/100 "
            f"(précédent {prev_score_txt}) | {item['reason']} | prix {r['price']:.2f} | "
            f"RSI {r['rsi']:.1f} | drawdown {r['drawdown_pct']:.1f}%"
        )
    return "\n".join(lines)


def send_email_alert(recipient_email, items):
    """Envoie l'alerte via l'API Resend ; aucun mot de passe de boîte mail n'est stocké."""
    if not RESEND_API_KEY or not RESEND_FROM_EMAIL:
        return False, "Configure RESEND_API_KEY et RESEND_FROM_EMAIL dans les secrets."
    payload = {
        "from": RESEND_FROM_EMAIL,
        "to": [recipient_email],
        "subject": f"📈 {len(items)} mouvement(s) sur ta watchlist - {date.today().isoformat()}",
        "html": _build_email_html(items),
        "text": _build_email_text(items),
    }
    try:
        response = requests.post(
            "https://api.resend.com/emails",
            headers={"Authorization": f"Bearer {RESEND_API_KEY}", "Content-Type": "application/json"},
            json=payload, timeout=20,
        )
        if response.ok:
            return True, None
        return False, f"Resend HTTP {response.status_code}: {response.text[:500]}"
    except Exception as exc:
        return False, str(exc)


# ============================================================
# CHATBOT : lecture rapide (règles) + analyse via l'API Claude
# ============================================================
def _fmt_pct(v, digits=1):
    return f"{v * 100:.{digits}f}%" if v is not None and pd.notna(v) else "N/A"


def build_stock_context(r: dict) -> str:
    """Résumé chiffré d'un titre, injecté dans le prompt du chatbot."""
    return (
        f"Ticker : {r['ticker']} ({r['name']}) - catégorie : {r['category']}\n"
        f"Prix : {r['price']:.2f}\n"
        f"Score global : {r['score']:.1f}/100 (seuil 'achat potentiel' {threshold_buy}, seuil 'à surveiller' {threshold_watch})\n"
        f"- RSI(14) : {r['rsi']:.1f} | variation 3 jours : {r['rsi_delta3']:+.1f} | points {r['rsi_score']:.1f}/{MAX_RSI}\n"
        f"- MACD histogramme : {r['macd_hist_pct']:+.2f}% du prix | pente 3 jours : {r['macd_slope_pct']:+.2f} | points {r['macd_score']:.1f}/{MAX_MACD}\n"
        f"- Volume (3 derniers jours / moyenne 20j) : x{r['vol_ratio']:.2f} | points {r['vol_score']:.1f}/{MAX_VOL}\n"
        f"- Baisse depuis le plus haut 6 mois : {r['drawdown_pct']:.1f}% | points {r['drawdown_score']:.1f}/{MAX_DRAWDOWN}\n"
        f"- Écart à la SMA50 : {r['pct_vs_sma50']:+.1f}% | points {r['support_score']:.1f}/{MAX_SUPPORT}\n"
        f"- Croissance CA (YoY) : {_fmt_pct(r['revenue_growth'])} | Marge brute : {_fmt_pct(r['gross_margin'])} "
        f"| Marge nette : {_fmt_pct(r['profit_margin'])}\n"
        f"- Points qualité : croissance {r['growth_score']:.1f}/{MAX_GROWTH}, marge brute {r['margin_score']:.1f}/{MAX_MARGIN}, "
        f"marge nette {r['profit_score']:.1f}/{MAX_PROFIT} (N/A = score neutre)\n"
        f"Données issues du dernier scan ({st.session_state.last_scan_time.strftime('%d/%m/%Y %H:%M') if st.session_state.last_scan_time else 'inconnu'})."
    )


def quick_read(r: dict):
    """Rule-based quick read; all explanatory text follows the selected UI language."""
    pos, neg = [], []
    if r["rsi"] < 35:
        pos.append(tr(f"RSI en zone de survente ({r['rsi']:.0f})", f"RSI in oversold territory ({r['rsi']:.0f})"))
    elif r["rsi"] > 65:
        neg.append(tr(f"RSI élevé ({r['rsi']:.0f}) : le titre a déjà fortement monté à court terme", f"High RSI ({r['rsi']:.0f}): the stock has already risen significantly in the short term"))
    if r["rsi_turn_score"] > 0 and r["rsi_delta3"] > 3:
        pos.append(tr(f"RSI qui rebondit (+{r['rsi_delta3']:.1f} pts en 3 jours) après être passé sous 40", f"RSI rebounding (+{r['rsi_delta3']:.1f} points in 3 days) after falling below 40"))
    if r["macd_bullish"] and r["macd_slope_pct"] > 0:
        pos.append(tr("MACD positif et en accélération", "Positive and accelerating MACD"))
    elif (not r["macd_bullish"]) and r["macd_slope_pct"] < 0:
        neg.append(tr("MACD négatif et en dégradation : momentum encore baissier", "Negative and deteriorating MACD: bearish momentum remains"))
    if r["vol_score"] >= 4:
        pos.append(tr(f"Volume supérieur à la normale (x{r['vol_ratio']:.2f}) sur un titre qui ne baisse pas", f"Above-normal volume (x{r['vol_ratio']:.2f}) while the stock is not falling"))
    elif r["vol_ratio"] < 0.8:
        neg.append(tr(f"Volume faible (x{r['vol_ratio']:.2f}) : peu de conviction du marché", f"Low volume (x{r['vol_ratio']:.2f}): limited market conviction"))
    if r["drawdown_pct"] >= 20:
        pos.append(tr(f"Décote de {r['drawdown_pct']:.0f}% par rapport au plus haut 6 mois", f"Trading {r['drawdown_pct']:.0f}% below its 6-month high"))
    elif r["drawdown_pct"] < 8:
        neg.append(tr(f"Peu de décote ({r['drawdown_pct']:.0f}%) : le point d'entrée n'est pas avantageux", f"Limited discount ({r['drawdown_pct']:.0f}%): the entry point may not be attractive"))
    if -3 <= r["pct_vs_sma50"] <= 3:
        pos.append(tr("Proche de la SMA50 : zone d'entrée classique", "Close to the SMA50: a common entry zone"))
    elif r["pct_vs_sma50"] > 12:
        neg.append(tr(f"Titre étendu ({r['pct_vs_sma50']:+.0f}% vs SMA50) : risque de repli", f"Price extended ({r['pct_vs_sma50']:+.0f}% vs SMA50): pullback risk"))
    elif r["pct_vs_sma50"] < -12:
        neg.append(tr(f"Nettement sous la SMA50 ({r['pct_vs_sma50']:+.0f}%) : tendance dégradée", f"Well below the SMA50 ({r['pct_vs_sma50']:+.0f}%): weakened trend"))
    if pd.notna(r["revenue_growth"]):
        if r["revenue_growth"] >= 0.15:
            pos.append(tr(f"Croissance du CA solide ({r['revenue_growth'] * 100:.0f}%)", f"Strong revenue growth ({r['revenue_growth'] * 100:.0f}%)"))
        elif r["revenue_growth"] < 0:
            neg.append(tr(f"Chiffre d'affaires en baisse ({r['revenue_growth'] * 100:.0f}%)", f"Revenue declining ({r['revenue_growth'] * 100:.0f}%)"))
    if pd.notna(r["profit_margin"]):
        if r["profit_margin"] < 0:
            neg.append(tr(f"Entreprise non rentable (marge nette {r['profit_margin'] * 100:.0f}%) : titre plus risqué", f"Company is unprofitable (net margin {r['profit_margin'] * 100:.0f}%): higher-risk stock"))
        elif r["profit_margin"] >= 0.10:
            pos.append(tr(f"Rentable (marge nette {r['profit_margin'] * 100:.0f}%)", f"Profitable (net margin {r['profit_margin'] * 100:.0f}%)"))

    color, _ = score_bucket(r["score"])
    verdict = {
        "green": tr("🟢 Configuration plutôt favorable à court terme", "🟢 Short-term setup looks favorable"),
        "yellow": tr("🟡 Neutre : mieux vaut attendre une confirmation", "🟡 Neutral: wait for confirmation"),
        "red": tr("🔴 Peu favorable à court terme", "🔴 Unfavorable in the short term"),
    }[color]
    return verdict, pos, neg


def translated_category(category: str) -> str:
    """Translate the categories used by the watchlist without changing stored values."""
    translations = {
        "Semi-conducteurs": "Semiconductors", "Logiciels": "Software", "Cybersécurité": "Cybersecurity",
        "Défense": "Defense", "Aérospatial": "Aerospace", "Énergie": "Energy",
        "Santé": "Healthcare", "Biotechnologies": "Biotechnology", "Finance": "Financials",
        "Consommation": "Consumer", "Industrie": "Industrials", "Automobile": "Automotive",
        "Matières premières": "Materials", "Télécommunications": "Telecommunications",
        "ETF": "ETF", "Diversifié": "Diversified", "Technologie": "Technology",
        "Big Tech": "Big Tech", "Biotech": "Biotech", "Consommation/Fintech": "Consumer/Fintech",
        "Croissance spéculative - Santé": "Speculative growth - Healthcare", "E-commerce": "E-commerce",
        "Fintech": "Fintech", "Fintech/SaaS": "Fintech/SaaS", "Logiciel/IA": "Software/AI",
        "Média": "Media", "Software/Cloud": "Software/Cloud", "Spatial/Défense": "Space/Defense",
        "Spéculatif - Batteries": "Speculative - Batteries", "Spéculatif - Biotech/IA": "Speculative - Biotech/AI",
        "Spéculatif - Biotechnologie": "Speculative - Biotechnology", "Spéculatif - IA": "Speculative - AI",
        "Spéculatif - IA/Défense": "Speculative - AI/Defense", "Spéculatif - IA/Santé": "Speculative - AI/Healthcare",
        "Spéculatif - Informatique quantique": "Speculative - Quantum computing", "Spéculatif - Nucléaire": "Speculative - Nuclear",
        "Spéculatif - Robotique": "Speculative - Robotics", "Spéculatif - Spatial": "Speculative - Space",
        "Spéculatif - Spatial/Données": "Speculative - Space/Data", "Spéculatif - Spatial/Défense": "Speculative - Space/Defense",
        "Spéculatif - eVTOL": "Speculative - eVTOL", "Énergie propre": "Clean energy", "Énergie solaire": "Solar energy",
    }
    return translations.get(str(category), str(category)) if IS_EN else str(category)


def render_stock_details(r: dict):
    """Expanded score breakdown and all the principal metrics for a selected stock."""
    color, status = score_bucket(r["score"])
    st.markdown(f"### {'📊 Stock details' if IS_EN else '📊 Détail de l’action'} — {r['ticker']} · {r['name']}")
    a, b, c, d = st.columns(4)
    a.metric(tr("Score global", "Overall score"), f"{r['score']:.1f}/100")
    b.metric(tr("Statut", "Status"), status)
    c.metric(tr("Prix actuel", "Current price"), f"{r['price']:.2f}")
    d.metric(tr("Catégorie", "Category"), translated_category(r.get("category", "—")))

    st.markdown(tr("#### Décomposition du score", "#### Score breakdown"))
    components = [
        (tr("RSI — niveau", "RSI — level"), r.get("rsi_level_score", 0), MAX_RSI_LEVEL, tr("Mesure le niveau de survente", "Measures oversold conditions")),
        (tr("RSI — retournement", "RSI — rebound"), r.get("rsi_turn_score", 0), MAX_RSI_TURN, tr("Récompense un rebond après une zone de survente", "Rewards a rebound after oversold conditions")),
        (tr("MACD — niveau", "MACD — level"), r.get("macd_level_score", 0), MAX_MACD_LEVEL, tr("Mesure le momentum actuel", "Measures current momentum")),
        (tr("MACD — accélération", "MACD — acceleration"), r.get("macd_slope_score", 0), MAX_MACD_SLOPE, tr("Mesure l’évolution récente du momentum", "Measures recent momentum change")),
        (tr("Volume", "Volume"), r.get("vol_score", 0), MAX_VOL, tr("Compare le volume récent à sa moyenne", "Compares recent volume with its average")),
        (tr("Baisse depuis le plus haut", "Drop from high"), r.get("drawdown_score", 0), MAX_DRAWDOWN, tr("Récompense une baisse depuis le plus haut sur six mois", "Rewards a pullback from the six-month high")),
        (tr("Proximité SMA50", "SMA50 proximity"), r.get("support_score", 0), MAX_SUPPORT, tr("Favorise un prix proche de la moyenne mobile à 50 jours", "Favors a price near the 50-day moving average")),
        (tr("Croissance du chiffre d’affaires", "Revenue growth"), r.get("growth_score", 0), MAX_GROWTH, tr("Croissance annuelle du chiffre d’affaires", "Year-over-year revenue growth")),
        (tr("Marge brute", "Gross margin"), r.get("margin_score", 0), MAX_MARGIN, tr("Rentabilité après coûts directs", "Profitability after direct costs")),
        (tr("Marge nette", "Net margin"), r.get("profit_score", 0), MAX_PROFIT, tr("Part du chiffre d’affaires conservée en bénéfice net", "Share of revenue retained as net profit")),
    ]
    for label, points, maximum, explanation in components:
        ratio = max(0, min(100, float(points) / maximum * 100 if maximum else 0))
        bar_color = "#4ade80" if ratio >= 70 else "#facc15" if ratio >= 40 else "#64748b"
        st.markdown(
            f"<div class='detail-row'><div><div class='detail-label'>{label}</div><div style='font-size:.72rem;color:#94a3b8;margin-top:2px'>{explanation}</div></div>"
            f"<div class='detail-track'><div class='stock-fill' style='width:{ratio:.1f}%;background:{bar_color}'></div></div>"
            f"<div class='detail-value'>{points:.1f}<span style='color:#94a3b8'>/{maximum}</span></div></div>",
            unsafe_allow_html=True,
        )

    st.markdown(tr("#### Indicateurs et données fondamentales", "#### Indicators and fundamentals"))
    def pct_or_na(value):
        return f"{value * 100:.1f}%" if value is not None and pd.notna(value) else "N/A"
    metrics = [
        ("RSI (14)", f"{r['rsi']:.1f}"),
        (tr("Variation RSI sur 3 jours", "3-day RSI change"), f"{r['rsi_delta3']:+.1f}"),
        ("MACD histogram", f"{r['macd_hist_pct']:+.3f}%"),
        (tr("Pente MACD sur 3 jours", "3-day MACD slope"), f"{r['macd_slope_pct']:+.3f}%"),
        (tr("Volume / moyenne 20 jours", "Volume / 20-day average"), f"{r['vol_ratio']:.2f}x"),
        (tr("Baisse depuis le plus haut 6 mois", "Drop from 6-month high"), f"{r['drawdown_pct']:.1f}%"),
        (tr("Écart à la SMA50", "Distance from SMA50"), f"{r['pct_vs_sma50']:+.1f}%"),
        (tr("Croissance du chiffre d’affaires", "Revenue growth"), pct_or_na(r.get("revenue_growth"))),
        (tr("Marge brute", "Gross margin"), pct_or_na(r.get("gross_margin"))),
        (tr("Marge nette", "Net margin"), pct_or_na(r.get("profit_margin"))),
    ]
    metric_cols = st.columns(2)
    for idx, (metric_name, metric_value) in enumerate(metrics):
        with metric_cols[idx % 2]:
            st.markdown(
                f"<div style='display:flex;justify-content:space-between;gap:12px;padding:10px 12px;margin:0 0 8px;border:1px solid #e5eaf3;border-radius:12px;background:linear-gradient(145deg,#ffffff,#f8faff)'>"
                f"<span style='color:#64748b;font-size:.82rem'>{metric_name}</span><strong style='color:#172033;font-size:.86rem;text-align:right'>{metric_value}</strong></div>",
                unsafe_allow_html=True,
            )
    st.caption(tr("Les données indisponibles sont affichées N/A et reçoivent un score fondamental neutre dans le modèle.", "Unavailable data is shown as N/A and receives a neutral fundamental score in this model."))


CHAT_SYSTEM_PROMPT = (
    "Tu es un analyste actions pragmatique et honnête, intégré à un scanner d'opportunités d'achat. "
    "Réponds dans la langue demandée par l'instruction de langue ajoutée au contexte. Sois concis et structuré. L'horizon est le COURT TERME (quelques jours à quelques semaines).\n"
    "Règles :\n"
    "- Appuie-toi sur les données chiffrées fournies (indicateurs techniques et fondamentaux). N'invente aucun chiffre.\n"
    "- Si la recherche web est disponible, utilise-la pour les actualités récentes (résultats, annonces, catalyseurs) et cite-les brièvement ; sinon dis que tu n'as pas accès aux actualités.\n"
    "- N’utilise pas de tableaux Markdown : présente les données en listes à puces courtes, avec les chiffres importants en gras. "
    "- Pour une analyse complète, utilise ce plan : **Verdict court terme** (Favorable / Neutre / Défavorable, une phrase), "
    "**Pourquoi** (3 à 5 points chiffrés), **Risques**, **Ce qu'il faut surveiller** (niveaux ou signaux qui confirmeraient ou invalideraient le scénario).\n"
    "- Explique le sens des indicateurs si c'est utile, sans jargon inutile. Signale les contradictions entre indicateurs.\n"
    "- Les indicateurs techniques ne prédisent pas l'avenir : exprime ton niveau de confiance. Rappelle en une ligne en fin d'analyse "
    "que ce n'est pas un conseil financier personnalisé.\n"
    f"Date du jour : {date.today().strftime('%d/%m/%Y')}."
)


def _groq_request(system: str, messages: list, use_web: bool):
    """Appelle Groq Chat Completions, avec recherche web intégrée en option."""
    api_messages = [{"role": "system", "content": system}] + messages
    body = {
        "model": GROQ_MODEL,
        "max_completion_tokens": 1500,
        "messages": api_messages,
    }
    if use_web:
        # Le modèle GPT-OSS de Groq prend en charge browser_search.
        body["tools"] = [{"type": "browser_search"}]
    return requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json",
        },
        json=body,
        timeout=90,
    )


def ask_groq(r: dict, history: list, use_web: bool):
    """Appelle Groq avec le contexte du titre. Renvoie (texte, erreur)."""
    if not GROQ_API_KEY:
        return None, "Configure GROQ_API_KEY dans les secrets Streamlit pour activer le chatbot."

    msgs = [{"role": m["role"], "content": m["content"]} for m in history[-12:]]
    while msgs and msgs[0]["role"] != "user":
        msgs.pop(0)

    language_instruction = (
        "Respond in English. Translate headings and explanations into natural English, while keeping ticker symbols and financial indicator names accurate."
        if IS_EN else
        "Réponds en français. Utilise des titres et explications naturels en français, en conservant les tickers et noms d'indicateurs financiers exacts."
    )
    system = CHAT_SYSTEM_PROMPT + "\n\n" + language_instruction + "\n\nSTOCK CONTEXT / DONNÉES DU TITRE SÉLECTIONNÉ :\n" + build_stock_context(r)
    if use_web:
        system += "\nLa recherche web est activée : utilise-la pour vérifier les actualités récentes pertinentes et cite brièvement les sources dans ta réponse."
    else:
        system += "\nLa recherche web est désactivée : ne prétends pas avoir vérifié les actualités en direct et précise cette limite si la question en dépend."

    try:
        resp = _groq_request(system, msgs, use_web)
        # Si browser_search échoue (modèle/outil indisponible), retenter sans outil web.
        if not resp.ok and use_web:
            system_without_web = system.replace(
                "La recherche web est activée : utilise-la pour vérifier les actualités récentes pertinentes et cite brièvement les sources dans ta réponse.",
                "La recherche web n'a pas fonctionné pour cette requête : ne prétends pas avoir vérifié les actualités en direct et précise cette limite si la question en dépend."
            )
            resp = _groq_request(system_without_web, msgs, False)
        if not resp.ok:
            return None, f"API Groq HTTP {resp.status_code}: {resp.text[:400]}"

        choices = resp.json().get("choices", [])
        answer = choices[0].get("message", {}).get("content", "").strip() if choices else ""
        return (answer or None), (None if answer else "Réponse vide de l'API Groq.")
    except Exception as exc:
        return None, f"Erreur lors de l'appel à Groq : {exc}"


def safe_md(text: str) -> str:
    """Évite que les '$' soient interprétés comme du LaTeX par Streamlit."""
    return text.replace("$", "\\$")


def chat_display_text(text: str) -> str:
    """Convertit les tableaux Markdown du modèle en listes lisibles."""
    lines = text.splitlines()
    output, table_rows = [], []

    def flush_table(rows):
        parsed = []
        for line in rows:
            cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
            if cells and all(re.fullmatch(r":?-{3,}:?", cell.replace(" ", "")) for cell in cells):
                continue
            if cells:
                parsed.append(cells)
        if not parsed:
            return []
        header = parsed[0]
        result = []
        for row in parsed[1:]:
            if len(row) == len(header):
                result.append("- " + " · ".join(f"**{header[i]} :** {value}" for i, value in enumerate(row) if value))
            else:
                result.append("- " + " · ".join(value for value in row if value))
        return result or ["- " + " · ".join(parsed[0])]

    for line in lines:
        if line.strip().startswith("|") and line.strip().endswith("|"):
            table_rows.append(line)
        else:
            if table_rows:
                output.extend(flush_table(table_rows))
                table_rows = []
            output.append(line)
    if table_rows:
        output.extend(flush_table(table_rows))
    return safe_md("\n".join(output))


# ============================================================
# SIDEBAR - CONFIG EMAIL & SCAN
# ============================================================
with st.sidebar:
    st.header(tr("⚙️ Configuration", "⚙️ Settings"))

    st.subheader(tr("📬 Alertes email", "📬 Email alerts"))
    email_enabled = st.checkbox(tr("Activer les alertes email", "Enable email alerts"), value=True)
    if "saved_recipient_email" not in st.session_state:
        st.session_state.saved_recipient_email = load_recipient_email()
    recipient_email = st.text_input(
        tr("Email destinataire", "Recipient email"),
        value=st.session_state.saved_recipient_email,
        placeholder="toi@example.com",
        key="recipient_email_input",
    )
    if st.button(tr("💾 Enregistrer mon email", "💾 Save my email"), use_container_width=True):
        email_to_save = recipient_email.strip()
        if "@" not in email_to_save or "." not in email_to_save.split("@")[-1]:
            st.error(tr("Entre une adresse email valide.", "Enter a valid email address."))
        else:
            ok, err = save_recipient_email(email_to_save)
            if ok:
                st.session_state.saved_recipient_email = email_to_save
                st.success(tr("Adresse enregistrée dans Supabase. Tu n'auras plus à la saisir à chaque lancement.", "Address saved to Supabase. You won't need to enter it each time."))
            else:
                st.error(err)
    if not supabase:
        st.warning(tr("Supabase non connecté : configure SUPABASE_URL et SUPABASE_KEY dans les secrets.", "Supabase is not connected: configure SUPABASE_URL and SUPABASE_KEY in your secrets."))
    if not RESEND_API_KEY or not RESEND_FROM_EMAIL:
        st.warning(tr("Configure aussi Resend : RESEND_API_KEY et RESEND_FROM_EMAIL.", "Also configure Resend: RESEND_API_KEY and RESEND_FROM_EMAIL."))
    else:
        st.caption(tr(f"Expéditeur configuré : {RESEND_FROM_EMAIL}", f"Configured sender: {RESEND_FROM_EMAIL}"))

    st.divider()
    st.subheader(tr("🔔 Déclenchement des alertes", "🔔 Alert triggers"))
    score_jump_threshold = st.slider(
        tr("Hausse de score jugée significative (points)", "Significant score increase (points)"), 5, 40, 15,
        help=tr("Une alerte part si le score d'un titre augmente d'au moins ce nombre de points depuis la dernière alerte, même sans changer de catégorie.", "An alert is sent if a stock's score rises by at least this many points since the previous alert, even if its category does not change.")
    )
    cooldown_minutes = st.slider(
        tr("Délai minimum entre 2 alertes pour un même titre (minutes)", "Minimum time between alerts for the same stock (minutes)"), 15, 720, 120,
        help=tr("Empêche le spam si un score oscille autour d'une frontière de catégorie.", "Prevents spam when a score fluctuates around a category boundary.")
    )
    st.caption(
        tr("Une alerte est envoyée quand un titre **change de catégorie** "
        "(🔴→🟡, 🟡→🟢, etc.) OU quand son score **progresse d'au moins "
        f"{score_jump_threshold} points**, avec un délai minimum de {cooldown_minutes} min par titre.", "An alert is sent when a stock **changes category** "
        "(🔴→🟡, 🟡→🟢, etc.) OR when its score **increases by at least "
        f"{score_jump_threshold} points**, with a {cooldown_minutes}-minute cooldown per stock." )
    )

    st.divider()
    st.subheader(tr("🎯 Seuils de score", "🎯 Score thresholds"))
    threshold_buy = st.slider(tr("Seuil 'Achat potentiel' 🟢", "'Potential buy' threshold 🟢"), 50, 100, 60)
    threshold_watch = st.slider(tr("Seuil 'À surveiller' 🟡", "'Watchlist' threshold 🟡"), 30, threshold_buy - 1, 42)

    st.divider()
    st.subheader(tr("🤖 Chatbot", "🤖 Chatbot"))
    chat_web_search = st.checkbox(tr("Autoriser la recherche web (actualités)", "Enable web search (news)"), value=True,
                                  help=tr("Utilise l'outil browser_search de Groq pour les actualités récentes. Si la recherche échoue, le chatbot réessaie sans.", "Uses Groq's browser_search tool for recent news. If the search fails, the chatbot retries without it."))
    if not GROQ_API_KEY:
        st.warning(tr("Configure GROQ_API_KEY dans les secrets pour activer le chatbot.", "Configure GROQ_API_KEY in your secrets to enable the chatbot."))
    else:
        st.caption(f"Modèle Groq : {GROQ_MODEL}")

    st.divider()
    st.subheader(tr("🔄 Scan automatique", "🔄 Automatic scan"))
    auto_scan = st.checkbox(tr("Rafraîchissement automatique", "Automatic refresh"), value=False, disabled=not AUTOREFRESH_AVAILABLE)
    if not AUTOREFRESH_AVAILABLE:
        st.caption(tr("Installe `streamlit-autorefresh` pour activer le scan automatique en continu.", "Install `streamlit-autorefresh` to enable continuous automatic scanning."))
    interval_min = st.number_input(tr("Intervalle (minutes)", "Interval (minutes)"), min_value=5, max_value=180, value=30, step=5)

    scan_button = st.button(tr("🔍 Scanner maintenant", "🔍 Scan now"), type="primary", use_container_width=True)

refresh_tick = 0
if AUTOREFRESH_AVAILABLE and auto_scan:
    refresh_tick = st_autorefresh(interval=int(interval_min) * 60 * 1000, key="auto_scanner_refresh")

# ============================================================
# ETAT SESSION
# ============================================================
if "ticker_alert_state" not in st.session_state:
    st.session_state.ticker_alert_state = load_persistent_alert_state()
if "last_results" not in st.session_state:
    st.session_state.last_results = None
if "last_scan_time" not in st.session_state:
    st.session_state.last_scan_time = None

if "last_refresh_tick" not in st.session_state:
    st.session_state.last_refresh_tick = refresh_tick
auto_triggered = bool(auto_scan and AUTOREFRESH_AVAILABLE and refresh_tick != st.session_state.last_refresh_tick)
st.session_state.last_refresh_tick = refresh_tick
should_scan = scan_button or auto_triggered or st.session_state.last_results is None


def score_bucket(score):
    if score >= threshold_buy:
        return "green", BUCKET_TEXT["green"]
    elif score >= threshold_watch:
        return "yellow", BUCKET_TEXT["yellow"]
    return "red", BUCKET_TEXT["red"]


BUCKET_RANK = {"red": 0, "yellow": 1, "green": 2}


# ============================================================
# SCAN
# ============================================================
if should_scan:
    results = []
    progress = st.progress(0.0, text=tr("Scan en cours...", "Scanning stocks..."))
    tickers = list(ALL_ASSETS.keys())

    for i, ticker in enumerate(tickers):
        res = analyze_ticker(ticker)
        if res:
            name, category = ALL_ASSETS[ticker]
            res["name"] = name
            res["category"] = category
            results.append(res)
        progress.progress((i + 1) / len(tickers), text=tr(f"Analyse de {ticker}...", f"Analyzing {ticker}..."))

    progress.empty()
    st.session_state.last_results = results
    st.session_state.last_scan_time = datetime.now()

    now = datetime.now()
    to_alert = []

    for r in results:
        ticker = r["ticker"]
        bucket_color, _ = score_bucket(r["score"])
        prev = st.session_state.ticker_alert_state.get(ticker)

        alert_reason = None
        prev_score = prev["score"] if prev else None
        prev_bucket = prev["bucket"] if prev else None

        if prev is not None:
            bucket_upgrade = BUCKET_RANK[bucket_color] > BUCKET_RANK[prev_bucket]
            score_jump = r["score"] - prev["score"]

            in_cooldown = False
            if prev.get("last_alert_time"):
                elapsed_min = (now - prev["last_alert_time"]).total_seconds() / 60
                in_cooldown = elapsed_min < cooldown_minutes

            if not in_cooldown:
                if bucket_upgrade:
                    alert_reason = (
                        f"Changement de catégorie : {BUCKET_TEXT[prev_bucket]} → {BUCKET_TEXT[bucket_color]}"
                    )
                elif score_jump >= score_jump_threshold:
                    alert_reason = f"Hausse de score significative : +{score_jump:.1f} pts"

        if alert_reason:
            to_alert.append({
                "result": r,
                "bucket": bucket_color,
                "reason": alert_reason,
                "prev_score": prev_score,
                "prev_bucket": prev_bucket,
            })

        new_state = {
            "score": r["score"],
            "bucket": bucket_color,
            "last_alert_time": now if alert_reason else (prev.get("last_alert_time") if prev else None),
        }
        st.session_state.ticker_alert_state[ticker] = new_state
        persist_alert_state(ticker, new_state)

    if to_alert and email_enabled:
        saved_email = (recipient_email or "").strip()
        if saved_email:
            ok, err = send_email_alert(saved_email, to_alert)
            if ok:
                st.toast(tr(f"📧 Email envoyé pour {len(to_alert)} mouvement(s)", f"📧 Email sent for {len(to_alert)} change(s)"))
            else:
                st.warning(tr(f"Échec de l'envoi de l'email : {err}", f"Email sending failed: {err}"))
        else:
            st.warning(tr("Enregistre d'abord ton adresse email dans la barre latérale.", "Save your email address in the sidebar first."))

results = st.session_state.last_results or []

# ============================================================
# HEADER RESUME
# ============================================================
FILTER_DEFAULTS = {
    "flt_search": "", "flt_cat": [], "flt_bucket": [], "flt_score": 0, "flt_rsi": 100, "flt_dd": 0,
    "flt_vol": 0.0, "flt_macd": False, "flt_sma": False, "flt_profit": False, "flt_rebound": False,
    "flt_sort": "Score", "flt_order": ("Descending" if IS_EN else "Décroissant"),
}
for _k, _v in FILTER_DEFAULTS.items():
    st.session_state.setdefault(_k, _v)


def reset_filters():
    for k, v in FILTER_DEFAULTS.items():
        st.session_state[k] = v


SORT_OPTIONS = ({
    "Score": "score", "RSI": "rsi", "Drop from high": "drawdown_pct",
    "Volume (ratio)": "vol_ratio", "Distance from SMA50": "pct_vs_sma50",
} if IS_EN else {
    "Score": "score", "RSI": "rsi", "Baisse depuis le plus haut": "drawdown_pct",
    "Volume (ratio)": "vol_ratio", "Écart à la SMA50": "pct_vs_sma50",
})
# Reset stale UI labels when the user switches language, preventing invalid widget states.
if st.session_state.get("flt_sort") not in SORT_OPTIONS:
    st.session_state.flt_sort = next(iter(SORT_OPTIONS))
valid_order_options = ["Descending", "Ascending"] if IS_EN else ["Décroissant", "Croissant"]
if st.session_state.get("flt_order") not in valid_order_options:
    st.session_state.flt_order = valid_order_options[0]
BUCKET_FILTER_LABELS = {BUCKET_TEXT["green"]: "green", BUCKET_TEXT["yellow"]: "yellow", BUCKET_TEXT["red"]: "red"}

if results:
    n_green = sum(1 for r in results if r["score"] >= threshold_buy)
    n_yellow = sum(1 for r in results if threshold_watch <= r["score"] < threshold_buy)
    n_red = len(results) - n_green - n_yellow

    c1, c2, c3, c4 = st.columns(4)
    c1.metric(tr("Actions suivies", "Tracked stocks"), len(results))
    c2.metric(tr("🟢 Achat potentiel", "🟢 Potential buy"), n_green)
    c3.metric(tr("🟡 À surveiller", "🟡 Watchlist"), n_yellow)
    c4.metric(tr("🔴 Rien à faire", "🔴 No action"), n_red)

    if st.session_state.last_scan_time:
        st.caption(tr(f"Dernier scan : {st.session_state.last_scan_time.strftime('%d/%m/%Y %H:%M:%S')}", f"Last scan: {st.session_state.last_scan_time.strftime('%Y-%m-%d %H:%M:%S')}"))

    df_results = pd.DataFrame(results).sort_values("score", ascending=False).reset_index(drop=True)
    df_results["bucket"] = df_results["score"].apply(lambda sc: score_bucket(sc)[0])
    categories = sorted(df_results["category"].unique())

    tab_overview, tab_chat = st.tabs([tr("📋 Vue d'ensemble", "📋 Overview"), tr("🤖 Chatbot", "🤖 Chatbot")])

    # ---------------- VUE D'ENSEMBLE + RECHERCHE PAR CRITÈRE ----------------
    with tab_overview:
        with st.expander(tr("🔎 Recherche & filtres", "🔎 Search & filters"), expanded=True):
            f1, f2, f3 = st.columns(3)
            f1.text_input(tr("Recherche (ticker ou nom)", "Search (ticker or name)"), key="flt_search", placeholder="e.g. NVDA, Palantir...")
            category_labels = {translated_category(cat): cat for cat in categories}
            if any(value not in category_labels for value in st.session_state.get("flt_cat", [])):
                st.session_state.flt_cat = []
            selected_category_labels = f2.multiselect(tr("Catégorie", "Category"), list(category_labels.keys()), key="flt_cat")
            f3.multiselect(tr("Statut", "Status"), list(BUCKET_FILTER_LABELS.keys()), key="flt_bucket")

            g1, g2, g3, g4 = st.columns(4)
            g1.slider(tr("Score minimum", "Minimum score"), 0, 100, key="flt_score")
            g2.slider(tr("RSI maximum", "Maximum RSI"), 0, 100, key="flt_rsi", help=tr("Ex : 35 pour ne garder que les titres en zone de survente.", "Example: 35 to show only stocks in oversold territory."))
            g3.slider(tr("Baisse minimale depuis le plus haut (%)", "Minimum drop from high (%)"), 0, 60, key="flt_dd")
            g4.slider(tr("Volume minimum (x moyenne 20j)", "Minimum volume (x 20-day average)"), 0.0, 3.0, step=0.1, key="flt_vol")

            h1, h2, h3, h4 = st.columns(4)
            h1.checkbox(tr("MACD haussier", "Bullish MACD"), key="flt_macd")
            h2.checkbox(tr("Au-dessus de la SMA50", "Above SMA50"), key="flt_sma")
            h3.checkbox(tr("Rentable (marge nette > 0)", "Profitable (net margin > 0)"), key="flt_profit")
            h4.checkbox(tr("RSI en rebond", "RSI rebounding"), key="flt_rebound", help=tr("RSI passé sous 40 récemment et qui remonte.", "RSI recently fell below 40 and is now rising."))

            s1, s2, s3 = st.columns([2, 2, 1])
            s1.selectbox(tr("Trier par", "Sort by"), list(SORT_OPTIONS.keys()), key="flt_sort")
            s2.radio(tr("Ordre", "Order"), ["Décroissant", "Croissant"] if not IS_EN else ["Descending", "Ascending"], horizontal=True, key="flt_order")
            s3.button(tr("↺ Réinitialiser", "↺ Reset"), on_click=reset_filters, use_container_width=True)

        fdf = df_results.copy()
        q = st.session_state.flt_search.strip().lower()
        if q:
            fdf = fdf[fdf["ticker"].str.lower().str.contains(q, regex=False) | fdf["name"].str.lower().str.contains(q, regex=False)]
        if selected_category_labels:
            selected_original_categories = [category_labels[label] for label in selected_category_labels]
            fdf = fdf[fdf["category"].isin(selected_original_categories)]
        if st.session_state.flt_bucket:
            fdf = fdf[fdf["bucket"].isin([BUCKET_FILTER_LABELS[b] for b in st.session_state.flt_bucket])]
        fdf = fdf[fdf["score"] >= st.session_state.flt_score]
        fdf = fdf[fdf["rsi"] <= st.session_state.flt_rsi]
        fdf = fdf[fdf["drawdown_pct"] >= st.session_state.flt_dd]
        fdf = fdf[fdf["vol_ratio"] >= st.session_state.flt_vol]
        if st.session_state.flt_macd:
            fdf = fdf[fdf["macd_bullish"]]
        if st.session_state.flt_sma:
            fdf = fdf[fdf["above_support"]]
        if st.session_state.flt_profit:
            fdf = fdf[fdf["profit_margin"].notna() & (fdf["profit_margin"] > 0)]
        if st.session_state.flt_rebound:
            fdf = fdf[fdf["rsi_turn_score"] > 0]
        fdf = fdf.sort_values(SORT_OPTIONS[st.session_state.flt_sort],
                              ascending=(st.session_state.flt_order == ("Ascending" if IS_EN else "Croissant")))

        st.caption(tr(f"{len(fdf)} résultat(s) sur {len(df_results)}", f"{len(fdf)} result(s) out of {len(df_results)}"))
        if fdf.empty:
            st.info(tr("Aucune action ne correspond à ces critères. Assouplis un filtre ou clique sur Réinitialiser.", "No stocks match these criteria. Loosen a filter or click Reset."))
        # Show details above the list so the panel is immediately visible after a click.
        selected_detail = st.session_state.get("selected_detail_ticker")
        available_tickers = set(df_results["ticker"].tolist())
        if selected_detail in available_tickers:
            detail_record = df_results[df_results["ticker"] == selected_detail].iloc[0].to_dict()
            with st.container(border=True):
                head_col, close_col = st.columns([6, 1])
                head_col.markdown(f"### {tr('Action sélectionnée', 'Selected stock')}: {selected_detail}")
                if close_col.button(tr("Fermer ✕", "Close ✕"), key="close_stock_details"):
                    st.session_state.selected_detail_ticker = None
                    st.rerun()
                render_stock_details(detail_record)
            st.divider()

        for _, r in fdf.iterrows():
            color, label = score_bucket(r["score"])
            score_pct = max(0, min(100, float(r["score"])))
            score_color = {"green": "#4ade80", "yellow": "#facc15", "red": "#fb7185"}.get(color, "#60a5fa")
            rsi_pct = max(0, min(100, float(r["rsi"])))
            rsi_color = "#4ade80" if 25 <= rsi_pct <= 40 else "#facc15" if rsi_pct < 55 else "#fb7185"
            vol_pct = max(0, min(100, float(r["vol_ratio"]) / 3 * 100))
            vol_color = "#4ade80" if r["vol_ratio"] >= 1.5 else "#facc15" if r["vol_ratio"] >= 1 else "#64748b"
            st.markdown(
                f"<div class='stock-card'><div class='stock-card-top'><div><div class='stock-ticker'>{r['ticker']} <span style='font-weight:500;color:#94a3b8'>·</span> {r['name']}</div>"
                f"<div class='stock-company'>{translated_category(r['category'])}</div></div>"
                f"<div style='text-align:right'><span class='score-badge badge-{color}'>{r['score']:.1f}/100</span><div style='font-size:.75rem;color:{score_color};margin-top:6px'>{label}</div></div></div>"
                f"<div class='stock-bars'>"
                f"<div><div class='stock-bar-label'><span>{tr('Score global', 'Overall score')}</span><strong style='color:#172033'>{score_pct:.0f}%</strong></div><div class='stock-track'><div class='stock-fill' style='width:{score_pct}%;background:{score_color}'></div></div></div>"
                f"<div><div class='stock-bar-label'><span>RSI (14)</span><strong style='color:#172033'>{r['rsi']:.0f}/100</strong></div><div class='stock-track'><div class='stock-fill' style='width:{rsi_pct}%;background:{rsi_color}'></div></div></div>"
                f"<div><div class='stock-bar-label'><span>{tr('Volume relatif', 'Relative volume')}</span><strong style='color:#172033'>{r['vol_ratio']:.2f}x</strong></div><div class='stock-track'><div class='stock-fill' style='width:{vol_pct}%;background:{vol_color}'></div></div></div></div>"
                f"<div class='stock-foot'><span>💵 {tr('Prix', 'Price')}: <strong style='color:#334155'>{r['price']:.2f}</strong></span>"
                f"<span>↘ {tr('Baisse 6 mois', '6-month drop')}: <strong style='color:#334155'>{r['drawdown_pct']:.1f}%</strong></span>"
                f"<span>📈 MACD: <strong style='color:{'#4ade80' if r['macd_bullish'] else '#fca5a5'}'>{tr('Haussier', 'Bullish') if r['macd_bullish'] else tr('Non haussier', 'Not bullish')}</strong></span></div></div>",
                unsafe_allow_html=True,
            )
            _, detail_col = st.columns([5, 1.2])
            detail_col.button(
                tr("Voir l'analyse complète ↗", "Open full analysis ↗"),
                key=f"details_{r['ticker']}",
                use_container_width=True,
                on_click=lambda ticker=r["ticker"]: st.session_state.update(selected_detail_ticker=ticker),
            )

    # ---------------- CHATBOT ----------------
    with tab_chat:
        st.markdown(tr("Choisis une action pour obtenir une lecture rapide, puis lance l'analyse court terme ou pose tes questions.", "Choose a stock for a quick read, then run a short-term analysis or ask questions."))
        labels = {t: f"{t} — {n} ({sc:.1f}/100)" for t, n, sc in zip(df_results["ticker"], df_results["name"], df_results["score"])}
        selected = st.selectbox(tr("Action", "Stock"), df_results["ticker"].tolist(), format_func=lambda t: labels[t], key="chat_ticker")
        r = df_results[df_results["ticker"] == selected].iloc[0].to_dict()
        color, label = score_bucket(r["score"])

        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric(tr("Score", "Score"), f"{r['score']:.1f}/100")
        m2.metric(tr("Statut", "Status"), label)
        m3.metric(tr("Prix", "Price"), f"{r['price']:.2f}")
        m4.metric("RSI", f"{r['rsi']:.0f}")
        m5.metric(tr("Baisse / plus haut", "Drop from high"), f"{r['drawdown_pct']:.1f}%")

        # A real graph is always shown in the chatbot tab, independently of the model's text format.
        with st.container(border=True):
            st.markdown(tr("#### 📈 Évolution du cours", "#### 📈 Price history"))
            price_history = fetch_history(selected)
            if not price_history.empty and "Close" in price_history.columns:
                price_series = price_history["Close"].dropna().tail(126)
                if isinstance(price_series, pd.DataFrame):
                    price_series = price_series.iloc[:, 0]
                if len(price_series) > 1:
                    chart_df = pd.DataFrame({tr("Cours", "Price"): pd.to_numeric(price_series, errors="coerce")}).dropna()
                    st.line_chart(chart_df, height=260, use_container_width=True)
                    first_price, last_price = float(chart_df.iloc[0, 0]), float(chart_df.iloc[-1, 0])
                    change_pct = ((last_price / first_price) - 1) * 100 if first_price else 0
                    st.caption(tr(
                        f"Environ 6 mois de séances · variation sur la période affichée : {change_pct:+.1f}%. Les performances passées ne préjugent pas des performances futures.",
                        f"About 6 months of trading sessions · change over the displayed period: {change_pct:+.1f}%. Past performance does not predict future results."
                    ))
                else:
                    st.info(tr("Pas assez de données pour tracer le graphique.", "Not enough data to plot a chart."))
            else:
                st.info(tr("Historique de cours indisponible pour ce titre.", "Price history is unavailable for this ticker."))

        verdict, pos, neg = quick_read(r)
        with st.expander(tr(f"Lecture rapide (règles) — {verdict}", f"Quick read (rules) — {verdict}"), expanded=True):
            p_col, n_col = st.columns(2)
            with p_col:
                st.markdown(tr("**✅ Points favorables**", "**✅ Positive signals**"))
                for item in pos or [tr("Aucun signal favorable marqué", "No strong positive signal")]:
                    st.markdown(f"- {item}")
            with n_col:
                st.markdown(tr("**⚠️ Points de vigilance**", "**⚠️ Watch-outs**"))
                for item in neg or [tr("Aucun signal défavorable marqué", "No strong negative signal")]:
                    st.markdown(f"- {item}")

        histories = st.session_state.setdefault("chat_histories", {})
        history = histories.setdefault(selected, [])

        b1, b2, _sp = st.columns([2, 2, 4])
        analyze_clicked = b1.button(tr("🧠 Analyse court terme", "🧠 Short-term analysis"), key=f"analyze_{selected}",
                                    disabled=not GROQ_API_KEY, use_container_width=True)
        if b2.button(tr("🗑️ Effacer la conversation", "🗑️ Clear conversation"), key=f"clear_{selected}", use_container_width=True):
            histories[selected] = []
            st.rerun()

        for msg in history:
            with st.chat_message(msg["role"]):
                st.markdown(chat_display_text(msg["content"]))

        user_prompt = st.chat_input(
            (tr(f"Pose une question sur {selected}...", f"Ask a question about {selected}...") if GROQ_API_KEY else tr("Chatbot indisponible : GROQ_API_KEY manquante", "Chatbot unavailable: GROQ_API_KEY is missing")),
            disabled=not GROQ_API_KEY, key="chat_input_box",
        )
        pending = None
        if analyze_clicked:
            pending = (
                f"Give me a short-term analysis of {selected}: is it a good idea to buy now? Explain why, the risks, and what to watch."
                if IS_EN else
                f"Donne-moi ton analyse court terme de {selected} : est-ce une bonne idée d'acheter maintenant ? Explique pourquoi, les risques et ce qu'il faut surveiller."
            )
        if user_prompt:
            pending = user_prompt

        if pending:
            history.append({"role": "user", "content": pending})
            with st.chat_message("user"):
                st.markdown(chat_display_text(pending))
            with st.chat_message("assistant"):
                with st.spinner(tr("Analyse en cours...", "Analyzing...")):
                    answer, err = ask_groq(r, history, chat_web_search)
                if answer:
                    st.markdown(chat_display_text(answer))
                    history.append({"role": "assistant", "content": answer})
                else:
                    st.error(err)
                    history.pop()
else:
    st.info(tr("Clique sur **Scanner maintenant** dans la barre latérale pour lancer la première analyse.", "Click **Scan now** in the sidebar to run your first analysis."))

st.divider()
st.caption(tr(
    "⚠️ La sauvegarde de l'adresse et de l'état des alertes est persistante via Supabase. "
    "Le scan automatique de cette interface fonctionne uniquement lorsque l'application reste active. "
    "Pour une surveillance 24/7 même lorsque l'application est fermée, il faut déployer un worker planifié séparé (par exemple GitHub Actions). "
    "Les analyses du chatbot sont informatives et ne constituent pas un conseil en investissement.",
    "⚠️ Email settings and alert states are stored persistently via Supabase. Automatic scanning works only while the app is active. "
    "For 24/7 monitoring while the app is closed, deploy a separate scheduled worker (for example, GitHub Actions). "
    "Chatbot analyses are for informational purposes and are not personalized investment advice."
))
