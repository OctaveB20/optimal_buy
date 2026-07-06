"""
📈 Stock Opportunity Scanner
----------------------------
Application Streamlit qui scanne une watchlist d'actions premium + les positions
du portefeuille, calcule un score d'opportunité d'achat (RSI, MACD, volume,
drawdown, support) et envoie un email automatique quand une action atteint
un score "optimal".

Scoring : chaque critère rapporte des points de façon CONTINUE (interpolation
linéaire entre une borne "faible" et une borne "excellente"), et non plus de
façon binaire (tout ou rien). Les bornes restent volontairement resserrées
pour garder une notation exigeante.

Installation :
    pip install streamlit yfinance pandas numpy streamlit-autorefresh

Lancement :
    streamlit run stock_scanner.py
"""

import time
from datetime import datetime, date

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

try:
    from streamlit_autorefresh import st_autorefresh
    AUTOREFRESH_AVAILABLE = True
except ImportError:
    AUTOREFRESH_AVAILABLE = False

# ============================================================
# CONFIG PAGE + STYLE (repris du look du portfolio tracker)
# ============================================================
st.set_page_config(page_title="Stock Opportunity Scanner", page_icon="📈", layout="wide")

st.markdown("""
<style>
    .main { background-color: #0e1117; }
    .block-container { padding-top: 2rem; }

    .card {
        background: linear-gradient(145deg, #161a23, #1c2029);
        border: 1px solid #2a2f3a;
        border-radius: 14px;
        padding: 16px 18px;
        margin-bottom: 14px;
    }
    .card-green { border-left: 5px solid #22c55e; }
    .card-yellow { border-left: 5px solid #eab308; }
    .card-red { border-left: 5px solid #ef4444; }

    .ticker-name { font-size: 1.05rem; font-weight: 700; color: #f5f5f5; }
    .ticker-sub { font-size: 0.78rem; color: #9aa0ab; margin-bottom: 6px; }

    .score-badge {
        display: inline-block;
        padding: 3px 12px;
        border-radius: 999px;
        font-weight: 700;
        font-size: 0.85rem;
    }
    .badge-green { background: rgba(34,197,94,0.18); color: #4ade80; }
    .badge-yellow { background: rgba(234,179,8,0.18); color: #facc15; }
    .badge-red { background: rgba(239,68,68,0.18); color: #f87171; }

    .crit-row { display: flex; justify-content: space-between; font-size: 0.82rem; }
    .crit-ok { color: #4ade80; }
    .crit-mid { color: #facc15; }
    .crit-ko { color: #6b7280; }

    .bar-track {
        background: #262b36;
        border-radius: 6px;
        height: 6px;
        width: 100%;
        margin: 2px 0 8px 0;
        overflow: hidden;
    }
    .bar-fill {
        height: 100%;
        border-radius: 6px;
    }

    .section-title {
        font-size: 1.15rem;
        font-weight: 700;
        color: #e5e7eb;
        margin: 22px 0 10px 0;
        border-bottom: 1px solid #2a2f3a;
        padding-bottom: 6px;
    }
</style>
""", unsafe_allow_html=True)

st.title("📈 Stock Opportunity Scanner")
st.caption("Détection automatique des meilleurs moments d'achat sur ta watchlist premium")

# ============================================================
# WATCHLIST (issue de ta liste "premium" + positions du portefeuille)
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
}

# Positions actuelles du portefeuille (hors ETF, non spéculatives et non déjà présentes)
PORTFOLIO_EXTRA = {
    "NVNO": ("enVVeno Medical", "Portefeuille - Spéculatif"),
    "OTLK": ("Outlook Therapeutics", "Portefeuille - Spéculatif"),
    "BNP.PA": ("BNP Paribas", "Portefeuille - Equity"),
}

ALL_ASSETS = {**WATCHLIST, **PORTFOLIO_EXTRA}

# ============================================================
# INDICATEURS TECHNIQUES
# ============================================================
def compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
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
    Interpolation linéaire non binaire entre deux bornes.
    - value <= worst (ou >= worst selon le sens)  -> 0 point
    - value atteint/dépasse best                  -> max_points
    - entre les deux                               -> proportionnel

    Fonctionne aussi bien pour un critère "plus c'est haut mieux c'est"
    (best > worst) que pour un critère "plus c'est bas mieux c'est"
    (best < worst, ex: RSI).
    """
    if best == worst:
        return max_points if value >= best else 0.0
    frac = (value - worst) / (best - worst)
    frac = max(0.0, min(1.0, frac))
    return round(frac * max_points, 2)


@st.cache_data(ttl=900, show_spinner=False)
def fetch_history(ticker: str) -> pd.DataFrame:
    try:
        df = yf.download(ticker, period="9mo", interval="1d", progress=False, auto_adjust=True)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df
    except Exception:
        return pd.DataFrame()


def analyze_ticker(ticker: str) -> dict | None:
    df = fetch_history(ticker)
    if df.empty or len(df) < 60 or "Close" not in df:
        return None

    close = df["Close"].dropna()
    volume = df["Volume"].dropna()

    # --- RSI ---
    rsi_series = compute_rsi(close)
    rsi = float(rsi_series.iloc[-1])
    # 30 pts max : RSI <= 20 (survente franche) -> 30 pts ; RSI >= 55 -> 0 pt
    rsi_score = scaled_score(rsi, worst=55, best=20, max_points=30)

    # --- MACD (force du signal, pas juste bullish/bearish) ---
    macd_line, signal_line = compute_macd(close)
    last_price = float(close.iloc[-1])
    macd_hist = float(macd_line.iloc[-1] - signal_line.iloc[-1])
    macd_hist_pct = (macd_hist / last_price) * 100 if last_price else 0.0
    macd_bullish = macd_hist > 0
    # 20 pts max : histogramme >= +1.0% du prix -> 20 pts ; <= -0.3% -> 0 pt
    macd_score = scaled_score(macd_hist_pct, worst=-0.3, best=1.0, max_points=20)

    # --- Volume ---
    vol_avg20 = volume.rolling(20).mean().iloc[-1]
    vol_today = volume.iloc[-1]
    vol_ratio = float(vol_today / vol_avg20) if pd.notna(vol_avg20) and vol_avg20 > 0 else 0.0
    vol_above_avg = vol_ratio > 1.0
    # 15 pts max : ratio >= 2.0x la moyenne -> 15 pts ; <= 0.8x -> 0 pt
    vol_score = scaled_score(vol_ratio, worst=0.8, best=2.0, max_points=15)

    # --- Drawdown ---
    recent_high = close.rolling(126, min_periods=30).max().iloc[-1]
    drawdown_pct = float((recent_high - last_price) / recent_high * 100) if recent_high else 0.0
    drawdown_ok = drawdown_pct > 15
    # 20 pts max : baisse >= 30% -> 20 pts ; <= 5% -> 0 pt
    drawdown_score = scaled_score(drawdown_pct, worst=5, best=30, max_points=20)

    # --- Support (position relative à la SMA50, en %) ---
    sma50 = close.rolling(50, min_periods=30).mean().iloc[-1]
    pct_vs_sma50 = float((last_price - sma50) / sma50 * 100) if pd.notna(sma50) and sma50 > 0 else -100.0
    above_support = pct_vs_sma50 > 0
    # 15 pts max : >= +8% au-dessus de la SMA50 -> 15 pts ; <= -5% -> 0 pt
    support_score = scaled_score(pct_vs_sma50, worst=-5, best=8, max_points=15)

    score = round(rsi_score + macd_score + vol_score + drawdown_score + support_score, 1)

    return {
        "ticker": ticker,
        "price": last_price,
        "rsi": rsi,
        "rsi_score": rsi_score,
        "macd_bullish": macd_bullish,
        "macd_hist_pct": macd_hist_pct,
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
        "score": score,
    }


# ============================================================
# EMAIL
# ============================================================
def send_email_alert(smtp_server, smtp_port, sender_email, sender_password, recipient_email, opportunities):
    msg = MIMEMultipart()
    msg["From"] = sender_email
    msg["To"] = recipient_email
    msg["Subject"] = f"🚀 {len(opportunities)} opportunité(s) d'achat détectée(s) - {date.today().isoformat()}"

    lines = ["Les actions suivantes ont atteint un score d'achat optimal :", ""]
    for opp in opportunities:
        lines.append(
            f"- {opp['ticker']} : score {opp['score']:.1f}/100 | RSI {opp['rsi']:.1f} | "
            f"prix {opp['price']:.2f} | drawdown {opp['drawdown_pct']:.1f}%"
        )
    msg.attach(MIMEText("\n".join(lines), "plain"))

    try:
        server = smtplib.SMTP(smtp_server, smtp_port, timeout=15)
        server.starttls()
        server.login(sender_email, sender_password)
        server.send_message(msg)
        server.quit()
        return True, None
    except Exception as e:
        return False, str(e)


# ============================================================
# SIDEBAR - CONFIG EMAIL & SCAN
# ============================================================
with st.sidebar:
    st.header("⚙️ Configuration")

    st.subheader("📬 Alertes email")
    email_enabled = st.checkbox("Activer les alertes email", value=False)
    smtp_server = st.text_input("Serveur SMTP", value="smtp.gmail.com")
    smtp_port = st.number_input("Port SMTP", value=587, step=1)
    sender_email = st.text_input("Email expéditeur")
    sender_password = st.text_input("Mot de passe / mot de passe d'application", type="password")
    recipient_email = st.text_input("Email destinataire")

    st.caption(
        "💡 Pour Gmail, utilise un **mot de passe d'application** "
        "(Compte Google → Sécurité → Mots de passe des applications), "
        "pas ton mot de passe habituel."
    )

    st.divider()
    st.subheader("🎯 Seuils de score")
    threshold_buy = st.slider("Seuil 'Achat potentiel' 🟢", 50, 100, 80)
    threshold_watch = st.slider("Seuil 'À surveiller' 🟡", 30, threshold_buy - 1, 60)

    st.divider()
    st.subheader("🔄 Scan automatique")
    auto_scan = st.checkbox("Rafraîchissement automatique", value=False, disabled=not AUTOREFRESH_AVAILABLE)
    if not AUTOREFRESH_AVAILABLE:
        st.caption("Installe `streamlit-autorefresh` pour activer le scan automatique en continu.")
    interval_min = st.number_input("Intervalle (minutes)", min_value=5, max_value=180, value=30, step=5)

    scan_button = st.button("🔍 Scanner maintenant", type="primary", use_container_width=True)

if AUTOREFRESH_AVAILABLE and auto_scan:
    st_autorefresh(interval=int(interval_min) * 60 * 1000, key="auto_scanner_refresh")

# ============================================================
# ETAT SESSION (pour éviter de spammer les emails)
# ============================================================
if "alerted_today" not in st.session_state:
    st.session_state.alerted_today = {}  # {ticker: date_str}
if "last_results" not in st.session_state:
    st.session_state.last_results = None
if "last_scan_time" not in st.session_state:
    st.session_state.last_scan_time = None

should_scan = scan_button or (auto_scan and AUTOREFRESH_AVAILABLE) or st.session_state.last_results is None


def score_bucket(score):
    if score >= threshold_buy:
        return "green", "🟢 Achat potentiel"
    elif score >= threshold_watch:
        return "yellow", "🟡 À surveiller"
    return "red", "🔴 Rien à faire"


def crit_class(points, max_points):
    """Couleur du critère selon la fraction de points obtenus (et non plus juste oui/non)."""
    frac = points / max_points if max_points else 0
    if frac >= 0.66:
        return "crit-ok"
    elif frac >= 0.33:
        return "crit-mid"
    return "crit-ko"


def bar_color(points, max_points):
    frac = points / max_points if max_points else 0
    if frac >= 0.66:
        return "#4ade80"
    elif frac >= 0.33:
        return "#facc15"
    return "#ef4444"


# ============================================================
# SCAN
# ============================================================
if should_scan:
    results = []
    progress = st.progress(0.0, text="Scan en cours...")
    tickers = list(ALL_ASSETS.keys())

    for i, ticker in enumerate(tickers):
        res = analyze_ticker(ticker)
        if res:
            name, category = ALL_ASSETS[ticker]
            res["name"] = name
            res["category"] = category
            results.append(res)
        progress.progress((i + 1) / len(tickers), text=f"Analyse de {ticker}...")

    progress.empty()
    st.session_state.last_results = results
    st.session_state.last_scan_time = datetime.now()

    # Détection des nouvelles opportunités (score >= seuil achat)
    today_str = date.today().isoformat()
    new_opportunities = [
        r for r in results
        if r["score"] >= threshold_buy
        and st.session_state.alerted_today.get(r["ticker"]) != today_str
    ]

    if new_opportunities and email_enabled:
        if sender_email and sender_password and recipient_email:
            ok, err = send_email_alert(
                smtp_server, int(smtp_port), sender_email, sender_password,
                recipient_email, new_opportunities
            )
            if ok:
                for r in new_opportunities:
                    st.session_state.alerted_today[r["ticker"]] = today_str
                st.toast(f"📧 Email envoyé pour {len(new_opportunities)} opportunité(s)")
            else:
                st.warning(f"Échec de l'envoi de l'email : {err}")
        else:
            st.warning("Renseigne les champs email dans la barre latérale pour activer les alertes.")

results = st.session_state.last_results or []

# ============================================================
# HEADER RESUME
# ============================================================
if results:
    n_green = sum(1 for r in results if r["score"] >= threshold_buy)
    n_yellow = sum(1 for r in results if threshold_watch <= r["score"] < threshold_buy)
    n_red = len(results) - n_green - n_yellow

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Actions suivies", len(results))
    c2.metric("🟢 Achat potentiel", n_green)
    c3.metric("🟡 À surveiller", n_yellow)
    c4.metric("🔴 Rien à faire", n_red)

    if st.session_state.last_scan_time:
        st.caption(f"Dernier scan : {st.session_state.last_scan_time.strftime('%d/%m/%Y %H:%M:%S')}")

    # ============================================================
    # AFFICHAGE PAR CATEGORIE (trié par score décroissant)
    # ============================================================
    df_results = pd.DataFrame(results).sort_values("score", ascending=False)
    categories = df_results["category"].unique()

    # Onglet vue d'ensemble + par catégorie
    tab_overview, tab_categories = st.tabs(["📋 Vue d'ensemble", "🗂️ Par catégorie"])

    with tab_overview:
        for _, r in df_results.iterrows():
            color, label = score_bucket(r["score"])
            cols = st.columns([3, 1, 1, 1, 1, 1])
            cols[0].markdown(f"**{r['ticker']}** — {r['name']}  \n<span style='color:#9aa0ab;font-size:0.8rem'>{r['category']}</span>", unsafe_allow_html=True)
            cols[1].markdown(f"<span class='score-badge badge-{color}'>{r['score']:.1f}/100</span>", unsafe_allow_html=True)
            cols[2].markdown(f"RSI: **{r['rsi']:.0f}**")
            cols[3].markdown(f"Prix: **{r['price']:.2f}**")
            cols[4].markdown(f"Drawdown: **{r['drawdown_pct']:.1f}%**")
            cols[5].markdown(label)
            st.divider()

    with tab_categories:
        for cat in sorted(categories):
            st.markdown(f"<div class='section-title'>{cat}</div>", unsafe_allow_html=True)
            cat_results = df_results[df_results["category"] == cat]
            cols = st.columns(3)
            for idx, (_, r) in enumerate(cat_results.iterrows()):
                color, label = score_bucket(r["score"])
                with cols[idx % 3]:
                    def bar(points, max_points):
                        pct = max(0, min(100, (points / max_points) * 100)) if max_points else 0
                        col = bar_color(points, max_points)
                        return f'<div class="bar-track"><div class="bar-fill" style="width:{pct:.0f}%; background:{col};"></div></div>'

                    st.markdown(f"""
                    <div class="card card-{color}">
                        <div class="ticker-name">{r['ticker']} <span class="score-badge badge-{color}">{r['score']:.1f}/100</span></div>
                        <div class="ticker-sub">{r['name']}</div>
                        <div style="font-size:0.85rem; line-height:1.5;">
                            <div class="crit-row"><span class="{crit_class(r['rsi_score'], 30)}">RSI {r['rsi']:.0f}</span><span>{r['rsi_score']:.1f}/30</span></div>
                            {bar(r['rsi_score'], 30)}
                            <div class="crit-row"><span class="{crit_class(r['macd_score'], 20)}">MACD hist. {r['macd_hist_pct']:+.2f}%</span><span>{r['macd_score']:.1f}/20</span></div>
                            {bar(r['macd_score'], 20)}
                            <div class="crit-row"><span class="{crit_class(r['vol_score'], 15)}">Volume x{r['vol_ratio']:.2f}</span><span>{r['vol_score']:.1f}/15</span></div>
                            {bar(r['vol_score'], 15)}
                            <div class="crit-row"><span class="{crit_class(r['drawdown_score'], 20)}">Baisse {r['drawdown_pct']:.1f}%</span><span>{r['drawdown_score']:.1f}/20</span></div>
                            {bar(r['drawdown_score'], 20)}
                            <div class="crit-row"><span class="{crit_class(r['support_score'], 15)}">Vs SMA50 {r['pct_vs_sma50']:+.1f}%</span><span>{r['support_score']:.1f}/15</span></div>
                            {bar(r['support_score'], 15)}
                            <b>Prix actuel : {r['price']:.2f}</b>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
else:
    st.info("Clique sur **Scanner maintenant** dans la barre latérale pour lancer la première analyse.")

st.divider()
st.caption(
    "⚠️ Le scan automatique fonctionne uniquement quand l'onglet Streamlit reste ouvert. "
    "Pour une surveillance 24/7 réelle, il faut planifier ce script (cron, tâche planifiée, "
    "ou un service cloud) en dehors de l'interface Streamlit."
)
