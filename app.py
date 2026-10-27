from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from stock_analysis import (
    DataValidationError,
    FEATURE_COLUMNS,
    chronological_split,
    evaluate_random_forest,
    prepare_labeled_data,
    validate_and_prepare_data,
)

HORIZON = 5
ACCENT = "#0f5b4d"
FONT_FAMILY = "Segoe UI, Arial, sans-serif"
st.session_state.setdefault("dark_theme", False)
IS_DARK = bool(st.session_state["dark_theme"])
THEME = (
    {
        "ink": "#e7f1ec",
        "muted": "#a9bdb4",
        "accent": "#8bd0b2",
        "mint": "#214b3e",
        "surface": "#17221e",
        "canvas": "#0f1714",
        "line": "#2d4038",
        "sidebar": "#0b221d",
        "sidebar_text": "#e7f1ec",
        "sidebar_muted": "#a9c9bc",
        "control": "#17221e",
        "control_hover": "#24483c",
        "shadow": "rgba(0, 0, 0, .2)",
    }
    if IS_DARK
    else {
        "ink": "#17342d",
        "muted": "#70817b",
        "accent": "#0f5b4d",
        "mint": "#dcefe8",
        "surface": "#ffffff",
        "canvas": "#f5f8f6",
        "line": "#e4ece8",
        "sidebar": "#123d34",
        "sidebar_text": "#e9f4ef",
        "sidebar_muted": "#b8d3c9",
        "control": "#ffffff",
        "control_hover": "#eaf5f0",
        "shadow": "rgba(29, 68, 55, .06)",
    }
)

st.set_page_config(page_title="PSX Intelligence", page_icon=None, layout="wide")

theme_css = """
    <style>
    :root {
        --display-font: "Segoe UI", Arial, sans-serif;
        --body-font: "Segoe UI", Arial, sans-serif;
        --ink: __INK__;
        --muted: __MUTED__;
        --accent: __ACCENT__;
        --mint: __MINT__;
        --surface: __SURFACE__;
        --canvas: __CANVAS__;
        --line: __LINE__;
        --sidebar: __SIDEBAR__;
        --sidebar-text: __SIDEBAR_TEXT__;
        --sidebar-muted: __SIDEBAR_MUTED__;
        --control: __CONTROL__;
        --control-hover: __CONTROL_HOVER__;
        --shadow: __SHADOW__;
    }
    .stApp, [data-testid="stAppViewContainer"] { background: var(--canvas); color: var(--ink); }
    [data-testid="stHeader"] { background: color-mix(in srgb, var(--canvas) 92%, transparent); }
    [data-testid="stSidebar"] {
        background: var(--sidebar);
        border-right: 0;
    }
    [data-testid="stSidebar"] * { color: var(--sidebar-text); }
    [data-testid="stSidebar"] .stFileUploader section {
        border: 1px dashed rgba(233,244,239,.38);
        background: rgba(255,255,255,.06);
    }
    .block-container { max-width: 1440px; padding: 5rem 3rem 4rem; }
    h1, h2, h3, h4, [data-testid="stMarkdownContainer"] { font-family: var(--display-font); color: var(--ink); letter-spacing: -0.03em; }
    p, label, .stCaption, [data-testid="stMetricLabel"], [data-testid="stWidgetLabel"] { font-family: var(--body-font); color: var(--ink); }
    .brand { display: flex; align-items: center; gap: .75rem; margin: .2rem 0 2.7rem; }
    .brand-mark { display: grid; place-items: center; width: 36px; height: 36px; border-radius: 11px; background: #b8e2d0; color: #123d34; font-family: var(--display-font); font-weight: 800; }
    .brand-name { font: 700 1.03rem var(--display-font); color: #f4fbf8; }
    .brand-sub { color: #9fc2b6; font: 500 .68rem var(--body-font); margin-top: .12rem; }
    .nav-label { color: #8db4a8; font: 700 .66rem var(--body-font); text-transform: uppercase; letter-spacing: .13em; margin: 1.4rem 0 .55rem; }
    .side-note { margin-top: 3rem; padding: .95rem; border: 1px solid rgba(216,238,228,.18); border-radius: 12px; color: var(--sidebar-muted); font: .72rem/1.5 var(--body-font); }
    .eyebrow { color: var(--accent); font: 700 .72rem var(--body-font); text-transform: uppercase; letter-spacing: .14em; margin-bottom: .45rem; }
    .hero-title { font: 800 2rem var(--display-font); color: var(--ink); letter-spacing: -.045em; margin: 0; }
    .hero-copy { color: var(--muted); font: .9rem/1.5 var(--body-font); margin-top: .35rem; }
    .topbar { display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap; gap: 1rem; margin-bottom: 2rem; }
    .topbar-copy { min-width: 0; max-width: 900px; }
    .hero-title { font-size: clamp(1.55rem, 3vw, 2rem); line-height: 1.2; overflow-wrap: anywhere; }
    .status-pill { display: inline-flex; align-items: center; gap: .45rem; padding: .45rem .75rem; background: color-mix(in srgb, var(--surface) 82%, var(--mint)); border: 1px solid var(--line); border-radius: 99px; color: var(--accent); font: 700 .72rem var(--body-font); }
    .status-dot { width: 7px; height: 7px; border-radius: 50%; background: #4ca77f; }
    .panel { background: var(--surface); border: 1px solid var(--line); border-radius: 16px; padding: 1.25rem 1.35rem; box-shadow: 0 8px 26px var(--shadow); }
    .panel-title { color: var(--ink); font: 700 .92rem var(--display-font); }
    .panel-kicker { color: var(--muted); font: .73rem var(--body-font); margin-top: .25rem; }
    .metric-card { background: var(--surface); border: 1px solid var(--line); border-radius: 14px; padding: 1rem 1.05rem; min-height: 100px; }
    .metric-label { color: var(--muted); font: 600 .71rem var(--body-font); }
    .metric-value { color: var(--ink); font: 800 1.35rem var(--display-font); margin-top: .42rem; }
    .metric-detail { color: var(--muted); font: .7rem var(--body-font); margin-top: .28rem; }
    .empty-wrap { display: grid; place-items: center; min-height: 385px; border: 1px dashed var(--line); border-radius: 18px; background: linear-gradient(135deg, var(--surface), color-mix(in srgb, var(--surface) 86%, var(--mint))); text-align: center; padding: 3rem; }
    .empty-mark { width: 58px; height: 58px; display: grid; place-items: center; border-radius: 18px; background: var(--mint); color: var(--accent); font: 800 1.4rem var(--display-font); margin: 0 auto 1rem; }
    .empty-title { color: var(--ink); font: 800 1.15rem var(--display-font); }
    .empty-copy { max-width: 480px; color: var(--muted); font: .82rem/1.55 var(--body-font); margin: .5rem auto 0; }
    .fine-print { color: var(--muted); font: .72rem/1.5 var(--body-font); }
    div[data-testid="stTabs"] button { font-family: var(--body-font); font-weight: 700; color: var(--muted); }
    div[data-testid="stTabs"] button[aria-selected="true"] { color: var(--accent); }
    div[data-testid="stFileUploader"] label, [data-testid="stFileUploaderDropzoneInstructions"] { font-weight: 700; color: var(--ink); }
    [data-testid="stFileUploader"] section, [data-testid="stFileUploaderDropzone"] { background: var(--control); border-color: var(--line); }
    [data-testid="stFileUploader"] small, [data-testid="stFileUploaderFileName"], [data-testid="stFileUploaderFileData"] { color: var(--muted); }
    [data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] * { color: var(--muted) !important; }
    [data-baseweb="popover"], [role="listbox"] { color: var(--ink); background: var(--surface); border-color: var(--line); }
    [role="option"], [data-baseweb="menu"] * { color: var(--ink); }
    .stAlert, [data-testid="stAlert"], [data-testid="stExpander"], [data-testid="stDataFrame"], [data-testid="stTable"] { border-radius: 12px; }
    [data-testid="stAlert"], [data-testid="stExpander"], [data-testid="stDataFrame"], [data-testid="stTable"] { border-color: var(--line); background: var(--surface); color: var(--ink); }
    [data-testid="stSelectbox"], [data-testid="stTextInput"], [data-testid="stNumberInput"], [data-testid="stDateInput"] { color: var(--ink); }
    [data-baseweb="select"] > div, input, textarea { background: var(--control) !important; color: var(--ink) !important; border-color: var(--line) !important; }
    input::placeholder, textarea::placeholder { color: var(--muted) !important; opacity: 1; }
    .stButton > button, [data-testid="stDownloadButton"] > button {
        color: var(--ink) !important; background: var(--control) !important; border: 1px solid var(--line) !important;
        border-radius: 9px !important; transition: background .15s ease, border-color .15s ease, transform .15s ease;
    }
    .stButton > button:hover, [data-testid="stDownloadButton"] > button:hover {
        color: var(--ink) !important; background: var(--control-hover) !important; border-color: var(--accent) !important;
    }
    .stButton > button:active, [data-testid="stDownloadButton"] > button:active { transform: translateY(1px); }
    .stButton > button:focus-visible, [data-testid="stDownloadButton"] > button:focus-visible {
        outline: 3px solid color-mix(in srgb, var(--accent) 35%, transparent) !important; outline-offset: 2px;
    }
    .stButton > button:disabled, [data-testid="stDownloadButton"] > button:disabled { color: var(--muted) !important; background: var(--canvas) !important; opacity: .6; }
    [data-testid="stFileUploaderDropzone"] { background: var(--control) !important; border-color: var(--line) !important; }
    [data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] button,
    [data-testid="stFileUploaderDropzone"] button,
    [data-testid="stFileUploader"] button {
        color: #ffffff !important; background: #0f5b4d !important; border: 1px solid #0f5b4d !important;
        border-radius: 8px !important; font-weight: 700 !important; transition: background .15s ease, border-color .15s ease;
    }
    [data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] button *,
    [data-testid="stFileUploaderDropzone"] button *,
    [data-testid="stFileUploader"] button * { color: #ffffff !important; fill: currentColor !important; }
    [data-testid="stFileUploaderDropzone"] button:hover, [data-testid="stFileUploader"] button:hover {
        color: #ffffff !important; background: #176b57 !important; border-color: #176b57 !important;
    }
    [data-testid="stFileUploaderDropzone"] button:active, [data-testid="stFileUploader"] button:active { background: #0b493c !important; }
    [data-testid="stFileUploaderDropzone"] button:focus-visible, [data-testid="stFileUploader"] button:focus-visible {
        outline: 3px solid #72c9a6 !important; outline-offset: 2px;
    }
    [data-testid="stFileUploaderDropzone"] button:disabled, [data-testid="stFileUploader"] button:disabled {
        color: #d1ded9 !important; background: #526b62 !important; border-color: #526b62 !important;
    }
    [data-testid="stRadio"] label { color: var(--sidebar-text) !important; }
    [data-testid="stProgressBar"] > div > div { background: var(--accent); }
    footer { background: var(--canvas); color: var(--muted); }
    .data-table { width: 100%; border-collapse: collapse; background: var(--surface); color: var(--ink); font: .78rem var(--body-font); }
    .data-table th, .data-table td { padding: .62rem .72rem; border-bottom: 1px solid var(--line); text-align: left; white-space: nowrap; }
    .data-table th { background: var(--mint); color: var(--ink); font-weight: 700; }
    .data-table-wrap { max-width: 100%; overflow-x: auto; border: 1px solid var(--line); border-radius: 12px; }
    @media (max-width: 900px) {
        .block-container { padding: 4.5rem 1.25rem 3rem; }
        .topbar { margin-bottom: 1.35rem; }
        .topbar-copy { width: 100%; }
        .hero-copy { max-width: 58ch; }
    }
    </style>
    """
for token, value in THEME.items():
    theme_css = theme_css.replace(f"__{token.upper()}__", value)
st.markdown(theme_css, unsafe_allow_html=True)


def sidebar() -> tuple[object, str]:
    with st.sidebar:
        st.markdown(
            """
            <div class="brand">
              <div class="brand-mark">PX</div>
              <div><div class="brand-name">PSX Intelligence</div><div class="brand-sub">Historical research workspace</div></div>
            </div>
            <div class="nav-label">Data source</div>
            """,
            unsafe_allow_html=True,
        )
        st.toggle("Dark theme", key="dark_theme")
        st.markdown(
            '<div class="nav-label">Workspace</div>',
            unsafe_allow_html=True,
        )
        section = st.radio(
            "Workspace section",
            ["Overview", "Model evaluation", "Data quality", "Methodology"],
            key="workspace_section",
            label_visibility="collapsed",
        )
        uploaded = st.file_uploader(
            "Upload historical CSV",
            type=["csv"],
            help="Required columns: Date, Open, High, Low, Close, Volume.",
        )
        st.markdown(
            """
            <div class="side-note"><strong>Research mode</strong><br>
            Local CSV analysis only. No live data, trading execution, or investment advice.</div>
            """,
            unsafe_allow_html=True,
        )
    return uploaded, section


def chart_layout(fig: go.Figure, height: int = 390) -> go.Figure:
    fig.update_layout(
        height=height,
        margin=dict(l=8, r=8, t=12, b=8),
        paper_bgcolor=THEME["surface"],
        plot_bgcolor=THEME["surface"],
        font=dict(family=FONT_FAMILY, color=THEME["ink"], size=11),
        hoverlabel=dict(
            bgcolor=THEME["surface"],
            font=dict(color=THEME["ink"], family=FONT_FAMILY),
        ),
        legend=dict(orientation="h", y=1.08, x=0, font=dict(size=10)),
        xaxis=dict(showgrid=False, zeroline=False, linecolor=THEME["line"]),
        yaxis=dict(
            showgrid=True,
            zeroline=False,
            gridcolor=THEME["line"],
            tickfont=dict(color=THEME["muted"]),
        ),
    )
    return fig


def price_figure(data: pd.DataFrame) -> go.Figure:
    plotted = data.copy()
    plotted["MA 20"] = plotted["Close"].rolling(20).mean()
    plotted["MA 50"] = plotted["Close"].rolling(50).mean()
    fig = go.Figure()
    close_color = "#62c99f" if IS_DARK else ACCENT
    ma20_color = "#a0d5bd" if IS_DARK else "#8ab9a8"
    ma50_color = "#e4bd78" if IS_DARK else "#b88737"
    fig.add_trace(go.Scatter(x=plotted["Date"], y=plotted["Close"], name="Close", mode="lines", line=dict(color=close_color, width=2.5)))
    fig.add_trace(go.Scatter(x=plotted["Date"], y=plotted["MA 20"], name="MA 20", mode="lines", line=dict(color=ma20_color, width=1.5)))
    fig.add_trace(go.Scatter(x=plotted["Date"], y=plotted["MA 50"], name="MA 50", mode="lines", line=dict(color=ma50_color, width=1.5)))
    fig.update_yaxes(title_text="Price")
    return chart_layout(fig)


def volume_figure(data: pd.DataFrame) -> go.Figure:
    up_color = "#62c99f" if IS_DARK else "#9bd2bc"
    down_color = "#e28f83" if IS_DARK else "#d9a29a"
    colors = [up_color if close >= open_ else down_color for close, open_ in zip(data["Close"], data["Open"])]
    fig = go.Figure(go.Bar(x=data["Date"], y=data["Volume"], name="Volume", marker_color=colors))
    fig.update_yaxes(title_text="Shares")
    return chart_layout(fig, 260)


def metric_card(label: str, value: str, detail: str) -> str:
    return f'<div class="metric-card"><div class="metric-label">{label}</div><div class="metric-value">{value}</div><div class="metric-detail">{detail}</div></div>'


uploaded, section = sidebar()
st.markdown(
    '<div class="topbar"><div class="topbar-copy"><div class="eyebrow">Pakistan Stock Exchange · Research workspace</div><div class="hero-title">Market intelligence, grounded in your data.</div><div class="hero-copy">Explore historical price behavior and evaluate an experimental directional model without live-market assumptions.</div></div><div class="status-pill"><span class="status-dot"></span> Local analysis</div></div>',
    unsafe_allow_html=True,
)

if not uploaded:
    if section == "Overview":
        st.markdown(
            '<div class="empty-wrap"><div><div class="empty-mark">+</div><div class="empty-title">Start with a historical dataset</div><div class="empty-copy">Upload a properly licensed daily OHLCV CSV from the sidebar to unlock price trends, volume context, data-quality checks, and chronological model evaluation.</div></div></div>',
            unsafe_allow_html=True,
        )
        st.markdown("<br>", unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        c1.markdown(metric_card("Required columns", "6", "Date + daily OHLCV"), unsafe_allow_html=True)
        c2.markdown(metric_card("Minimum history", "100", "Valid daily rows"), unsafe_allow_html=True)
        c3.markdown(metric_card("Forecast horizon", "5 days", "Experimental estimate"), unsafe_allow_html=True)
    elif section == "Model evaluation":
        st.markdown('<div class="panel"><div class="panel-title">Model evaluation</div><div class="panel-kicker">Upload valid historical OHLCV data to compute a chronological holdout evaluation and baseline comparison.</div></div>', unsafe_allow_html=True)
    elif section == "Data quality":
        st.markdown('<div class="panel"><div class="panel-title">Data quality</div><div class="panel-kicker">Upload a CSV to inspect accepted rows, dates, required fields, and validation results.</div></div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="panel"><div class="panel-title">Methodology</div><div class="panel-kicker">The dashboard validates OHLCV data, creates historical features, then evaluates the Random Forest on a later chronological holdout separated by a forecast-horizon gap. A majority-class baseline is shown for context. All estimates are experimental, not financial advice.</div></div>', unsafe_allow_html=True)
    st.markdown(
        '<p class="fine-print" style="margin-top:1.5rem;">Research-only workspace. No live market feed is connected, and no values are fabricated before a file is uploaded.</p>',
        unsafe_allow_html=True,
    )
    st.stop()

try:
    raw = pd.read_csv(uploaded)
    data = validate_and_prepare_data(raw)
    labeled = prepare_labeled_data(data, horizon=HORIZON)
    train, test = chronological_split(labeled, horizon=HORIZON)
    evaluation = evaluate_random_forest(train, test)
except DataValidationError as exc:
    st.error(str(exc))
    st.stop()
except pd.errors.EmptyDataError:
    st.error("The uploaded CSV is empty.")
    st.stop()
except (pd.errors.ParserError, UnicodeDecodeError) as exc:
    st.error(f"Could not read this CSV. Check its encoding and structure: {exc}")
    st.stop()

latest_close = float(data.iloc[-1]["Close"])
previous_close = float(data.iloc[-2]["Close"])
change = latest_close - previous_close
change_pct = change / previous_close if previous_close else 0
latest_features = labeled.iloc[[-1]]
probability_up = float(
    evaluation.model.predict_proba(latest_features[FEATURE_COLUMNS])[
        0, list(evaluation.model.classes_).index(1)
    ]
)

st.markdown(
    "".join(
        [
            metric_card("Latest close", f"{latest_close:,.2f}", f"{change:+,.2f} · {change_pct:+.2%} vs prior session"),
            metric_card("History window", f"{len(data):,} rows", f"{data['Date'].min():%d %b %Y} — {data['Date'].max():%d %b %Y}"),
            metric_card("Model estimate", f"{probability_up:.1%}", f"Probability of higher close in {HORIZON} sessions"),
            metric_card("Holdout balance", f"{evaluation.metrics['Balanced accuracy']:.1%}", f"{len(test):,} later rows held out"),
        ]
    ),
    unsafe_allow_html=True,
)
st.markdown("<br>", unsafe_allow_html=True)

if section == "Overview":
    left, right = st.columns([1.7, 1], gap="large")
    with left:
        st.markdown('<div class="panel"><div class="panel-title">Price action</div><div class="panel-kicker">Close with 20- and 50-session moving averages</div></div>', unsafe_allow_html=True)
        st.plotly_chart(price_figure(data), use_container_width=True, theme="streamlit", config={"displaylogo": False, "modeBarButtonsToRemove": ["lasso2d", "select2d"]})
    with right:
        st.markdown('<div class="panel"><div class="panel-title">Directional estimate</div><div class="panel-kicker">Experimental model output, not a trading signal</div></div>', unsafe_allow_html=True)
        st.markdown(f'<div style="padding:1.7rem 0 .8rem;"><div class="eyebrow">Higher close in {HORIZON} sessions</div><div style="font:800 2.65rem var(--display-font);color:{THEME["accent"]};">{probability_up:.1%}</div><div class="fine-print">Based only on the uploaded historical file.</div></div>', unsafe_allow_html=True)
        st.progress(probability_up)
        st.info("Historical estimate only. It is not financial advice or a guarantee of future results.")
        st.markdown(f'<div class="fine-print">Training period: {len(train):,} rows · Test period: {len(test):,} rows</div>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="panel"><div class="panel-title">Volume context</div><div class="panel-kicker">Session volume colored by close direction</div></div>', unsafe_allow_html=True)
    st.plotly_chart(volume_figure(data), use_container_width=True, theme="streamlit", config={"displaylogo": False})

if section == "Model evaluation":
    st.markdown('<div class="panel"><div class="panel-title">Chronological holdout results</div><div class="panel-kicker">The model only sees earlier rows; later rows remain untouched until evaluation.</div></div>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    model_cols = st.columns(4)
    for col, (name, value) in zip(model_cols, evaluation.metrics.items()):
        col.markdown(metric_card(name, f"{value:.1%}", "Random Forest · test set"), unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="panel"><div class="panel-title">Baseline comparison</div><div class="panel-kicker">A majority-class baseline provides a simple reference point for model usefulness.</div></div>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    baseline_cols = st.columns(4)
    for col, (name, value) in zip(baseline_cols, evaluation.baseline_metrics.items()):
        col.markdown(metric_card(name, f"{value:.1%}", "Majority class · test set"), unsafe_allow_html=True)
    st.caption("Metrics are historical estimates. They can change across market regimes and exclude transaction costs, slippage, taxes, dividends, and corporate actions.")
    importance = pd.DataFrame({"Feature": FEATURE_COLUMNS, "Importance": evaluation.model.feature_importances_}).sort_values("Importance", ascending=True)
    feature_color = "#62c99f" if IS_DARK else ACCENT
    fig = go.Figure(go.Bar(x=importance["Importance"], y=importance["Feature"], orientation="h", marker_color=feature_color))
    fig.update_layout(title="Exploratory feature importance", height=360, margin=dict(l=8, r=8, t=42, b=8))
    chart_layout(fig, 360)
    st.plotly_chart(fig, use_container_width=True, theme="streamlit", config={"displaylogo": False})

if section == "Data quality":
    st.markdown('<div class="panel"><div class="panel-title">Data quality review</div><div class="panel-kicker">Checks performed before any features or model fitting are created.</div></div>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    quality_cols = st.columns(4)
    quality_cols[0].markdown(metric_card("Rows accepted", f"{len(data):,}", "After sorting and de-duplication"), unsafe_allow_html=True)
    quality_cols[1].markdown(metric_card("Unique sessions", f"{data['Date'].nunique():,}", "One row per date"), unsafe_allow_html=True)
    quality_cols[2].markdown(metric_card("Feature rows", f"{len(labeled):,}", "Enough history and future label"), unsafe_allow_html=True)
    quality_cols[3].markdown(metric_card("Missing values", "0", "In required OHLCV columns"), unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        f'<div class="data-table-wrap">{data.tail(25).to_html(index=False, escape=True, classes="data-table")}</div>',
        unsafe_allow_html=True,
    )
    st.caption("The validator checks required columns, parseable dates, finite numeric values, positive prices, non-negative volume, valid OHLC relationships, minimum history, and duplicate dates.")

if section == "Methodology":
    st.markdown('<div class="panel"><div class="panel-title">How this workspace works</div><div class="panel-kicker">A transparent research workflow designed to be easy to inspect.</div></div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div style="padding:1.3rem 0;">
        <h4>1. Validate</h4><p class="fine-print">The uploaded table is checked for the required Date, Open, High, Low, Close, and Volume columns. Invalid dates, non-finite values, impossible OHLC relationships, negative volume, duplicate dates, and insufficient history are surfaced before analysis.</p>
        <h4>2. Create historical features</h4><p class="fine-print">Returns, rolling volatility, moving-average ratios, volume change, and daily range are calculated from information available on or before each session.</p>
        <h4>3. Evaluate chronologically</h4><p class="fine-print">Rows are kept in time order. The earlier period trains the Random Forest, a horizon-sized gap separates train and test labels, and the later period is used for holdout metrics. A majority-class baseline is shown for context.</p>
        <h4>4. Interpret carefully</h4><p class="fine-print">The latest probability is an experimental estimate from the uploaded file. It is not a forecast guarantee, buy/sell instruction, or financial advice. Historical accuracy does not establish future performance.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
