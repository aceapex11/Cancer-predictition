import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# ── PAGE CONFIG ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="OncoPulse · Survival Intelligence",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── DARK THEME CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

*, *::before, *::after { box-sizing: border-box; }

html, body, [class*="css"], .stApp {
    background-color: #080C10 !important;
    color: #C8D4E0 !important;
    font-family: 'Syne', sans-serif !important;
}

/* ── sidebar ── */
[data-testid="stSidebar"] {
    background: #0C1118 !important;
    border-right: 1px solid #1C2A38 !important;
}
[data-testid="stSidebar"] > div { padding-top: 0 !important; }
[data-testid="stSidebarContent"] { padding: 0 !important; }

/* ── main content ── */
.main .block-container {
    padding: 2rem 2.5rem 3rem !important;
    max-width: 1300px !important;
}

/* ── hide chrome ── */
#MainMenu, footer, header { visibility: hidden; }
[data-testid="stDecoration"] { display: none; }
[data-testid="stToolbar"] { display: none; }

/* ── headings ── */
h1, h2, h3, h4 { font-family: 'Syne', sans-serif !important; color: #E8F0F8 !important; }

/* ── radio buttons (nav) ── */
[data-testid="stRadio"] > div { gap: 2px !important; }
[data-testid="stRadio"] label {
    display: flex !important;
    align-items: center !important;
    gap: 8px !important;
    padding: 9px 14px !important;
    border-radius: 8px !important;
    font-size: 13px !important;
    color: #6B8299 !important;
    cursor: pointer !important;
    transition: all 0.15s !important;
    font-family: 'Syne', sans-serif !important;
}
[data-testid="stRadio"] label:hover { background: #111920 !important; color: #C8D4E0 !important; }
[data-testid="stRadio"] input[type="radio"]:checked + div { color: #00D4FF !important; }

/* ── inputs ── */
[data-testid="stNumberInput"] input,
[data-testid="stSelectbox"] > div > div {
    background: #0F1922 !important;
    border: 1px solid #1C2A38 !important;
    border-radius: 8px !important;
    color: #C8D4E0 !important;
    font-family: 'Syne', sans-serif !important;
    font-size: 13px !important;
}
[data-testid="stNumberInput"] input:focus,
[data-testid="stSelectbox"] > div > div:focus {
    border-color: #00D4FF !important;
    box-shadow: 0 0 0 2px rgba(0,212,255,0.1) !important;
}

/* ── labels ── */
[data-testid="stNumberInput"] label,
[data-testid="stSelectbox"] label,
[data-testid="stRadio"] > label {
    color: #4A6278 !important;
    font-size: 11px !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.08em !important;
    font-family: 'Syne', sans-serif !important;
}

/* ── buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #00D4FF 0%, #0099CC 100%) !important;
    color: #060A0D !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 600 !important;
    font-size: 13px !important;
    padding: 0.6rem 1.6rem !important;
    letter-spacing: 0.04em !important;
    transition: all 0.2s !important;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #33DDFF 0%, #00BBEE 100%) !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 20px rgba(0,212,255,0.3) !important;
}

/* ── metric cards ── */
[data-testid="metric-container"] {
    background: #0C1520 !important;
    border: 1px solid #1C2A38 !important;
    border-radius: 10px !important;
    padding: 1rem 1.25rem !important;
}
[data-testid="stMetricValue"] { color: #E8F0F8 !important; font-family: 'Syne', sans-serif !important; }
[data-testid="stMetricLabel"] { color: #4A6278 !important; font-size: 11px !important; }

/* ── divider ── */
hr { border-color: #1C2A38 !important; }

/* ── info box ── */
[data-testid="stAlert"] {
    background: #0A1520 !important;
    border: 1px solid #1C3A50 !important;
    border-radius: 8px !important;
    color: #6BBFDD !important;
}

/* ── tabs ── */
[data-testid="stTabs"] [role="tablist"] {
    border-bottom: 1px solid #1C2A38 !important;
    gap: 0 !important;
}
[data-testid="stTabs"] [role="tab"] {
    background: transparent !important;
    color: #4A6278 !important;
    border: none !important;
    border-bottom: 2px solid transparent !important;
    font-family: 'Syne', sans-serif !important;
    font-size: 13px !important;
    padding: 8px 18px !important;
    border-radius: 0 !important;
}
[data-testid="stTabs"] [role="tab"][aria-selected="true"] {
    color: #00D4FF !important;
    border-bottom: 2px solid #00D4FF !important;
    background: transparent !important;
}

/* ── custom classes ── */
.onco-card {
    background: #0C1520;
    border: 1px solid #1C2A38;
    border-radius: 12px;
    padding: 1.25rem 1.5rem;
    margin-bottom: 1rem;
}
.onco-label {
    font-size: 10px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #4A6278;
    margin-bottom: 0.6rem;
}
.banner-alive {
    background: rgba(16,60,20,0.6);
    border: 1px solid #1E5C28;
    border-radius: 10px;
    padding: 14px 20px;
    font-size: 15px;
    font-weight: 600;
    color: #4ADB6A;
    margin-bottom: 1rem;
    font-family: 'Syne', sans-serif;
}
.banner-dead {
    background: rgba(60,16,16,0.6);
    border: 1px solid #5C1E1E;
    border-radius: 10px;
    padding: 14px 20px;
    font-size: 15px;
    font-weight: 600;
    color: #FF5555;
    margin-bottom: 1rem;
    font-family: 'Syne', sans-serif;
}
.patient-table { width:100%; border-collapse:collapse; font-size:13px; font-family:'Syne',sans-serif; }
.patient-table td { padding:7px 4px; border-bottom:1px solid #121E28; }
.patient-table td:first-child { color:#4A6278; }
.patient-table td:last-child { font-weight:500; text-align:right; color:#C8D4E0; }
.section-sep {
    font-size: 10px; font-weight: 700; text-transform: uppercase;
    letter-spacing: 0.12em; color: #4A6278;
    margin: 1.5rem 0 1rem; padding-bottom: 0.5rem;
    border-bottom: 1px solid #1C2A38;
}
</style>
""", unsafe_allow_html=True)

# ── DARK PLOTLY THEME ──────────────────────────────────────────────────────────
DARK_BG    = "#080C10"
CARD_BG    = "#0C1520"
GRID_COLOR = "#1C2A38"
TEXT_COLOR = "#C8D4E0"
CYAN       = "#00D4FF"
GREEN      = "#4ADB6A"
RED        = "#FF5555"
AMBER      = "#FFB347"
PURPLE     = "#BB86FC"
PINK       = "#FF79C6"

PLOTLY_LAYOUT = dict(
    paper_bgcolor=CARD_BG,
    plot_bgcolor=CARD_BG,
    font=dict(family="Syne, sans-serif", color=TEXT_COLOR, size=12),
    margin=dict(t=40, b=40, l=50, r=20),
)
AXIS_STYLE = dict(gridcolor=GRID_COLOR, zerolinecolor=GRID_COLOR, color=TEXT_COLOR)

MODEL_COLORS = {
    "Logistic Regression": CYAN,
    "Decision Tree":       "#FF9F43",
    "Random Forest":       GREEN,
    "KNN":                 PURPLE,
    "SVM":                 PINK,
    "Naive Bayes":         AMBER,
    "Gradient Boosting":   "#5BC0EB",
    "XGBoost":             "#E84393",
}


def chart_model_status_table(results):
    models  = [r["model"]      for r in results]
    preds   = [r["prediction"] for r in results]
    alive_p = [r["alive_pct"]  for r in results]
    dead_p  = [r["dead_pct"]   for r in results]

    pred_colors  = ["#4ADB6A" if p == "Alive" else "#FF5555" for p in preds]
    row_bg_even  = ["#0C1520"] * len(models)
    row_bg_odd   = ["#0A1218"] * len(models)
    row_bgs      = [row_bg_odd[i] if i % 2 == 0 else row_bg_even[i] for i in range(len(models))]

    alive_bars = []
    dead_bars  = []
    for ap, dp in zip(alive_p, dead_p):
        if ap is not None:
            alive_bar = (
                f"<span style=\'display:inline-block;width:{ap}%;max-width:100%;"
                f"background:#4ADB6A;height:8px;border-radius:4px;\'></span> {ap}%"
            )
            dead_bar = (
                f"<span style=\'display:inline-block;width:{dp}%;max-width:100%;"
                f"background:#FF5555;height:8px;border-radius:4px;\'></span> {dp}%"
            )
        else:
            alive_bar = "N/A"
            dead_bar  = "N/A"
        alive_bars.append(alive_bar)
        dead_bars.append(dead_bar)

    mc = [MODEL_COLORS.get(m, "#888") for m in models]

    fig = make_subplots(
        rows=1, cols=2,
        column_widths=[0.48, 0.52],
        specs=[[{"type": "table"}, {"type": "bar"}]],
        horizontal_spacing=0.04,
    )

    fig.add_trace(go.Table(
        header=dict(
            values=["<b>Model</b>", "<b>Status</b>", "<b>Alive %</b>", "<b>Dead %</b>"],
            fill_color="#0A1830",
            font=dict(color=["#C8D4E0"]*4, size=11, family="Syne, sans-serif"),
            align=["left","center","center","center"],
            line_color="#1C2A38",
            height=36,
        ),
        cells=dict(
            values=[
                [f"<b>{m}</b>" for m in models],
                preds,
                [f"{v}%" if v is not None else "N/A" for v in alive_p],
                [f"{v}%" if v is not None else "N/A" for v in dead_p],
            ],
            fill_color=[
                ["#0C1520"]*len(models),
                [("rgba(16,60,20,0.6)" if p=="Alive" else "rgba(60,16,16,0.6)") for p in preds],
                ["#0C1520"]*len(models),
                ["#0C1520"]*len(models),
            ],
            font=dict(
                color=[mc, pred_colors, ["#4ADB6A"]*len(models), ["#FF5555"]*len(models)],
                size=12,
                family="Syne, sans-serif",
            ),
            align=["left","center","center","center"],
            line_color="#1C2A38",
            height=34,
        ),
    ), row=1, col=1)

    alive_vals = [v if v is not None else 0 for v in alive_p]
    dead_vals  = [v if v is not None else 0 for v in dead_p]

    fig.add_trace(go.Bar(
        name="Alive %", x=models, y=alive_vals,
        marker=dict(color=["#4ADB6A"]*len(models), opacity=0.85),
        text=[f"{v}%" if v else "N/A" for v in alive_p],
        textposition="outside", textfont=dict(size=10, color="#4ADB6A"),
    ), row=1, col=2)

    fig.add_trace(go.Bar(
        name="Dead %", x=models, y=dead_vals,
        marker=dict(color=["#FF5555"]*len(models), opacity=0.75),
        text=[f"{v}%" if v else "N/A" for v in dead_p],
        textposition="outside", textfont=dict(size=10, color="#FF5555"),
    ), row=1, col=2)

    fig.update_layout(
        paper_bgcolor=CARD_BG, plot_bgcolor=CARD_BG,
        font=dict(family="Syne, sans-serif", color=TEXT_COLOR, size=12),
        height=420,
        margin=dict(t=50, b=60, l=10, r=20),
        barmode="group",
        title=dict(text="All models — status & probability breakdown",
                   font=dict(size=14, color=TEXT_COLOR)),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=TEXT_COLOR),
                    orientation="h", x=0.55, y=1.08),
        xaxis2=dict(gridcolor=GRID_COLOR, color=TEXT_COLOR, tickangle=-30,
                    tickfont=dict(size=9)),
        yaxis2=dict(gridcolor=GRID_COLOR, color=TEXT_COLOR, range=[0, 120],
                    title="Probability (%)"),
    )
    return fig

# ── LOAD MODEL ─────────────────────────────────────────────────────────────────
@st.cache_resource
def load_bundle():
    return joblib.load("breast_cancer_models_compressed.pkl")

bundle = load_bundle()
scaler = bundle["scaler"]

MODEL_NAMES = list(MODEL_COLORS.keys())

# ── SYNTHETIC REFERENCE DATA (for distribution plots) ─────────────────────────
@st.cache_data
def get_reference_data():
    rng = np.random.default_rng(42)
    n = 800
    alive_mask = rng.random(n) > 0.3
    df = pd.DataFrame({
        "Status":         np.where(alive_mask, "Alive", "Dead"),
        "Survival_Months": np.clip(
            np.where(alive_mask,
                     rng.normal(72, 18, n),
                     rng.normal(38, 15, n)), 1, 107).astype(int),
        "Tumor_Size":      np.clip(
            np.where(alive_mask,
                     rng.normal(25, 12, n),
                     rng.normal(40, 18, n)), 1, 140).astype(int),
        "Regional_Nodes":  np.clip(
            np.where(alive_mask,
                     rng.normal(3, 3, n),
                     rng.normal(8, 5, n)), 1, 46).astype(int),
        "Age":             np.clip(
            np.where(alive_mask,
                     rng.normal(55, 12, n),
                     rng.normal(62, 10, n)), 25, 95).astype(int),
    })
    return df

REF = get_reference_data()

# ── HELPERS ────────────────────────────────────────────────────────────────────
def build_input_df(sm, rn, ts, es, ps, ast):
    return pd.DataFrame([[
        sm, rn,
        0 if es  == "Positive" else 1,
        0 if ps  == "Positive" else 1,
        ts,
        0 if ast == "Regional" else 1,
    ]], columns=["Survival Months","Reginol Node Positive",
                 "Estrogen Status","Progesterone Status",
                 "Tumor Size","A Stage"])

def get_proba(model, scaled):
    has_proba = hasattr(model, "predict_proba") and (
        getattr(model, "probability", True) is not False)
    if has_proba:
        p = model.predict_proba(scaled)[0]
        return round(p[0]*100,1), round(p[1]*100,1)
    return None, None

def run_all_models(scaled):
    results = []
    for name in MODEL_NAMES:
        m    = bundle[name]
        pred = m.predict(scaled)[0]
        ap, dp = get_proba(m, scaled)
        results.append({"model": name, "prediction": "Alive" if pred==0 else "Dead",
                         "alive_pct": ap, "dead_pct": dp})
    return results

# ── CHARTS ─────────────────────────────────────────────────────────────────────
def chart_proba_bars(results):
    names  = [r["model"] for r in results if r["alive_pct"] is not None]
    alive  = [r["alive_pct"] for r in results if r["alive_pct"] is not None]
    dead   = [r["dead_pct"]  for r in results if r["dead_pct"]  is not None]
    colors = [MODEL_COLORS[n] for n in names]
    fig = go.Figure()
    fig.add_trace(go.Bar(name="Alive %", x=names, y=alive,
                         marker_color=[GREEN]*len(names), opacity=0.85,
                         text=[f"{v}%" for v in alive], textposition="outside",
                         textfont=dict(size=11)))
    fig.add_trace(go.Bar(name="Dead %", x=names, y=dead,
                         marker_color=[RED]*len(names), opacity=0.75,
                         text=[f"{v}%" for v in dead], textposition="outside",
                         textfont=dict(size=11)))
    fig.update_layout(**PLOTLY_LAYOUT,
        title=dict(text="Alive vs Dead probability — all models", font=dict(size=14, color=TEXT_COLOR)),
        barmode="group", height=360,
        yaxis=dict(**AXIS_STYLE, range=[0,120], title="Probability (%)"),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=TEXT_COLOR)))
    return fig

def chart_alive_gauge_dots(results):
    names  = [r["model"] for r in results if r["alive_pct"] is not None]
    alive  = [r["alive_pct"] for r in results if r["alive_pct"] is not None]
    colors = [GREEN if v >= 50 else RED for v in alive]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=alive, y=names, mode="markers+text",
        marker=dict(size=22, color=colors, line=dict(width=2, color="#1C2A38")),
        text=[f"{v}%" for v in alive], textposition="middle center",
        textfont=dict(size=10, color="#060A0D", family="Syne"),
    ))
    fig.add_vline(x=50, line_dash="dash", line_color=AMBER, line_width=1.5,
                  annotation_text="50% threshold", annotation_font_color=AMBER,
                  annotation_position="top right")
    fig.update_layout(**PLOTLY_LAYOUT,
        title=dict(text="Alive probability per model", font=dict(size=14, color=TEXT_COLOR)),
        height=320, xaxis=dict(**AXIS_STYLE, range=[0,110], title="Alive probability (%)"),
        yaxis=dict(**AXIS_STYLE, title=""))
    return fig

def chart_radar(results):
    proba_models = [r for r in results if r["alive_pct"] is not None]
    names  = [r["model"] for r in proba_models]
    values = [r["alive_pct"] for r in proba_models]
    values_closed = values + [values[0]]
    names_closed  = names + [names[0]]
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values_closed, theta=names_closed,
        fill="toself", fillcolor="rgba(0,212,255,0.12)",
        line=dict(color=CYAN, width=2),
        marker=dict(size=6, color=CYAN),
    ))
    fig.update_layout(
        paper_bgcolor=CARD_BG, plot_bgcolor=CARD_BG,
        font=dict(family="Syne", color=TEXT_COLOR, size=11),
        polar=dict(
            bgcolor=CARD_BG,
            radialaxis=dict(visible=True, range=[0,100], gridcolor=GRID_COLOR,
                            color=TEXT_COLOR, tickfont=dict(size=9)),
            angularaxis=dict(gridcolor=GRID_COLOR, color=TEXT_COLOR, tickfont=dict(size=10)),
        ),
        title=dict(text="Model agreement radar", font=dict(size=14, color=TEXT_COLOR)),
        height=380, margin=dict(t=60, b=40, l=60, r=60),
        showlegend=False,
    )
    return fig

def chart_violin_survival(patient_sm):
    fig = go.Figure()
    for status, color in [("Alive", GREEN), ("Dead", RED)]:
        sub = REF[REF["Status"] == status]["Survival_Months"]
        fig.add_trace(go.Violin(
            y=sub, name=status, box_visible=True,
            meanline_visible=True, fillcolor=color.replace("FF","44"),
            line_color=color, opacity=0.85,
        ))
    fig.add_hline(y=patient_sm, line_dash="dot", line_color=CYAN, line_width=2,
                  annotation_text=f"Patient: {patient_sm}mo",
                  annotation_font_color=CYAN, annotation_position="top right")
    fig.update_layout(**PLOTLY_LAYOUT,
        title=dict(text="Survival months distribution — Alive vs Dead", font=dict(size=14, color=TEXT_COLOR)),
        height=360, yaxis=dict(**AXIS_STYLE, title="Survival months"),
        xaxis=dict(**AXIS_STYLE, title=""),
        violingap=0.3, violinmode="overlay")
    return fig

def chart_violin_tumor(patient_ts):
    fig = go.Figure()
    for status, color in [("Alive", GREEN), ("Dead", RED)]:
        sub = REF[REF["Status"] == status]["Tumor_Size"]
        fig.add_trace(go.Violin(
            y=sub, name=status, box_visible=True,
            meanline_visible=True, fillcolor=color.replace("FF","44"),
            line_color=color, opacity=0.85,
        ))
    fig.add_hline(y=patient_ts, line_dash="dot", line_color=AMBER, line_width=2,
                  annotation_text=f"Patient tumor: {patient_ts}mm",
                  annotation_font_color=AMBER, annotation_position="top right")
    fig.update_layout(**PLOTLY_LAYOUT,
        title=dict(text="Tumor size distribution — Alive vs Dead", font=dict(size=14, color=TEXT_COLOR)),
        height=360, yaxis=dict(**AXIS_STYLE, title="Tumor size (mm)"),
        xaxis=dict(**AXIS_STYLE, title=""),
        violingap=0.3, violinmode="overlay")
    return fig

def chart_nodes_scatter(patient_rn, patient_ts):
    fig = go.Figure()
    for status, color in [("Alive", GREEN), ("Dead", RED)]:
        sub = REF[REF["Status"] == status]
        fig.add_trace(go.Scatter(
            x=sub["Regional_Nodes"], y=sub["Tumor_Size"],
            mode="markers", name=status,
            marker=dict(size=5, color=color, opacity=0.45,
                        line=dict(width=0)),
        ))
    fig.add_trace(go.Scatter(
        x=[patient_rn], y=[patient_ts], mode="markers+text",
        name="Patient", text=["◀ Patient"],
        textposition="middle right", textfont=dict(color=CYAN, size=12),
        marker=dict(size=16, color=CYAN, symbol="star",
                    line=dict(width=2, color="#fff")),
    ))
    fig.update_layout(**PLOTLY_LAYOUT,
        title=dict(text="Regional nodes vs tumor size — patient position", font=dict(size=14, color=TEXT_COLOR)),
        height=360, xaxis=dict(**AXIS_STYLE, title="Regional nodes positive"),
        yaxis=dict(**AXIS_STYLE, title="Tumor size (mm)"),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=TEXT_COLOR)))
    return fig

def chart_risk_heatmap(results):
    proba_models = [r for r in results if r["alive_pct"] is not None]
    names  = [r["model"] for r in proba_models]
    alive  = [r["alive_pct"] for r in proba_models]
    dead   = [r["dead_pct"]  for r in proba_models]
    z = [alive, dead]
    fig = go.Figure(go.Heatmap(
        z=z, x=names, y=["Alive %","Dead %"],
        colorscale=[[0,"#1a0a0a"],[0.5,"#3a2020"],[1,RED]],
        text=[[f"{v}%" for v in row] for row in z],
        texttemplate="%{text}", textfont=dict(size=11, color="white"),
        showscale=False,
    ))
    fig.update_layout(**PLOTLY_LAYOUT,
        title=dict(text="Probability heatmap across all models", font=dict(size=14, color=TEXT_COLOR)),
        height=220, margin=dict(t=50, b=60, l=80, r=20),
        xaxis=dict(**AXIS_STYLE, tickangle=-30))
    return fig

def chart_consensus_donut(alive_count, dead_count):
    fig = go.Figure(go.Pie(
        values=[alive_count, dead_count],
        labels=["Alive","Dead"],
        hole=0.65,
        marker=dict(colors=[GREEN, RED]),
        textinfo="label+percent",
        textfont=dict(size=12, color=TEXT_COLOR),
    ))
    fig.update_layout(
        paper_bgcolor=CARD_BG, plot_bgcolor=CARD_BG,
        font=dict(family="Syne", color=TEXT_COLOR),
        height=280, margin=dict(t=20, b=20, l=20, r=20),
        showlegend=False,
        annotations=[dict(text=f"{alive_count}/{alive_count+dead_count}<br><span style='font-size:10px'>agree</span>",
                          x=0.5, y=0.5, font_size=18, font_color=TEXT_COLOR,
                          showarrow=False)]
    )
    return fig

def chart_survival_months_hist(patient_sm):
    fig = go.Figure()
    for status, color in [("Alive", GREEN), ("Dead", RED)]:
        sub = REF[REF["Status"]==status]["Survival_Months"]
        fig.add_trace(go.Histogram(
            x=sub, name=status, nbinsx=20,
            marker_color=color, opacity=0.6,
        ))
    fig.add_vline(x=patient_sm, line_dash="dot", line_color=CYAN, line_width=2,
                  annotation_text=f"Patient", annotation_font_color=CYAN,
                  annotation_position="top right")
    fig.update_layout(**PLOTLY_LAYOUT, barmode="overlay",
        title=dict(text="Survival months histogram", font=dict(size=14, color=TEXT_COLOR)),
        height=300, xaxis=dict(**AXIS_STYLE, title="Months"),
        yaxis=dict(**AXIS_STYLE, title="Count"),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=TEXT_COLOR)))
    return fig

def chart_node_violin():
    fig = go.Figure()
    for status, color in [("Alive", GREEN), ("Dead", RED)]:
        sub = REF[REF["Status"]==status]["Regional_Nodes"]
        fig.add_trace(go.Violin(
            y=sub, name=status, box_visible=True,
            meanline_visible=True, fillcolor=color.replace("FF","44"),
            line_color=color, opacity=0.9,
        ))
    fig.update_layout(**PLOTLY_LAYOUT,
        title=dict(text="Regional nodes positive — Alive vs Dead", font=dict(size=14, color=TEXT_COLOR)),
        height=320, yaxis=dict(**AXIS_STYLE, title="Nodes positive"),
        violingap=0.3, violinmode="overlay",
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=TEXT_COLOR)))
    return fig

# ── SIDEBAR ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding:1.5rem 1.25rem 1.25rem;border-bottom:1px solid #1C2A38;margin-bottom:0.75rem;">
        <div style="display:flex;align-items:center;gap:10px;margin-bottom:4px;">
            <div style="width:34px;height:34px;border-radius:8px;background:#00243A;
                        border:1px solid #004060;display:flex;align-items:center;
                        justify-content:center;font-size:17px;">🧬</div>
            <div>
                <div style="font-weight:700;font-size:15px;color:#E8F0F8;
                            font-family:'Syne',sans-serif;letter-spacing:0.02em;">OncoPulse</div>
                <div style="font-size:10px;color:#4A6278;letter-spacing:0.08em;
                            text-transform:uppercase;">Survival Intelligence</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    page = st.radio("Navigation", ["🩺  Predict", "📊  Compare Models", "📈  Data Explorer"],
                    label_visibility="collapsed")

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — PREDICT
# ═══════════════════════════════════════════════════════════════════════════════
if page == "🩺  Predict":
    st.markdown("<h2 style='margin-bottom:0.25rem;'>Patient survival prediction</h2>", unsafe_allow_html=True)
    st.markdown("<p style='color:#4A6278;font-size:13px;margin-bottom:1.5rem;'>Select a model, enter clinical data, run prediction.</p>", unsafe_allow_html=True)

    top_col1, top_col2 = st.columns([2, 1])
    with top_col1:
        model_name = st.selectbox("Model", MODEL_NAMES)
    with top_col2:
        st.markdown("<br>", unsafe_allow_html=True)
        run_btn = st.button("⚡  Run prediction", use_container_width=True)

    st.markdown('<div class="section-sep">Clinical inputs</div>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        survival_months = st.number_input("Survival months", 1, 107, 60)
        estrogen_status = st.selectbox("Estrogen status", ["Positive","Negative"])
    with c2:
        regional_nodes = st.number_input("Regional nodes positive", 1, 46, 5)
        progesterone_status = st.selectbox("Progesterone status", ["Positive","Negative"])
    with c3:
        tumor_size = st.number_input("Tumor size (mm)", 1, 140, 30)
        a_stage = st.selectbox("A stage", ["Regional","Distant"])

    if run_btn:
        input_df = build_input_df(survival_months, regional_nodes, tumor_size,
                                  estrogen_status, progesterone_status, a_stage)
        scaled   = scaler.transform(input_df)
        all_results = run_all_models(scaled)
        model    = bundle[model_name]
        pred     = model.predict(scaled)[0]
        ap, dp   = get_proba(model, scaled)

        st.markdown('<div class="section-sep">Result</div>', unsafe_allow_html=True)

        if pred == 0:
            st.markdown('<div class="banner-alive">✦ Predicted status: ALIVE</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="banner-dead">✦ Predicted status: DEAD</div>', unsafe_allow_html=True)

        m1, m2, m3 = st.columns(3)
        m1.metric("Alive probability", f"{ap}%" if ap is not None else "N/A")
        m2.metric("Dead probability",  f"{dp}%" if dp is not None else "N/A")
        m3.metric("Model used", model_name.split()[0])

        if ap is None:
            st.info("SVM was trained with probability=False — only the class prediction is available.")

        r1, r2, r3 = st.columns(3)
        with r1:
            # Probability bar
            if ap is not None:
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=["Alive","Dead"], y=[ap, dp],
                    marker_color=[GREEN, RED],
                    text=[f"{ap}%", f"{dp}%"],
                    textposition="outside", textfont=dict(size=13, color=TEXT_COLOR),
                    width=0.45,
                ))
                fig.update_layout(**PLOTLY_LAYOUT,
                    height=300, showlegend=False,
                    title=dict(text="Prediction confidence", font=dict(size=13, color=TEXT_COLOR)),
                    yaxis=dict(**AXIS_STYLE, range=[0,120], showticklabels=False, showgrid=False),
                    xaxis=dict(**AXIS_STYLE, showgrid=False),
                )
                st.plotly_chart(fig, use_container_width=True)

        with r2:
            # Pie chart — alive vs dead for selected model
            if ap is not None:
                model_color = MODEL_COLORS.get(model_name, CYAN)
                pie_fig = go.Figure(go.Pie(
                    values=[ap, dp],
                    labels=["Alive", "Dead"],
                    hole=0.55,
                    marker=dict(
                        colors=[GREEN, RED],
                        line=dict(color=CARD_BG, width=3),
                    ),
                    textinfo="label+percent",
                    textfont=dict(size=12, color=TEXT_COLOR, family="Syne, sans-serif"),
                    direction="clockwise",
                    pull=[0.05 if pred == 0 else 0, 0.05 if pred == 1 else 0],
                ))
                pie_fig.update_layout(
                    paper_bgcolor=CARD_BG, plot_bgcolor=CARD_BG,
                    font=dict(family="Syne, sans-serif", color=TEXT_COLOR),
                    height=300,
                    margin=dict(t=50, b=20, l=20, r=20),
                    showlegend=False,
                    title=dict(
                        text=f"{model_name} — survival split",
                        font=dict(size=13, color=TEXT_COLOR),
                    ),
                    annotations=[dict(
                        text=f"{'ALIVE' if pred==0 else 'DEAD'}",
                        x=0.5, y=0.5,
                        font=dict(size=15, color=GREEN if pred==0 else RED,
                                  family="Syne, sans-serif"),
                        showarrow=False,
                    )],
                )
                st.plotly_chart(pie_fig, use_container_width=True)
            else:
                st.markdown(
                    "<div style='height:300px;display:flex;align-items:center;justify-content:center;"
                    "color:#4A6278;font-size:13px;border:1px solid #1C2A38;border-radius:10px;'>"
                    "Pie chart unavailable — SVM has no probability output</div>",
                    unsafe_allow_html=True
                )

        with r3:
            # Patient summary
            st.markdown('<div class="onco-label" style="margin-top:0.5rem;">Patient record</div>', unsafe_allow_html=True)
            rows = [("Model", model_name), ("Survival months", survival_months),
                    ("Regional nodes", regional_nodes), ("Tumor size", f"{tumor_size} mm"),
                    ("Estrogen", estrogen_status), ("Progesterone", progesterone_status),
                    ("A stage", a_stage)]
            html = '<table class="patient-table">'
            for k, v in rows:
                html += f"<tr><td>{k}</td><td>{v}</td></tr>"
            html += "</table>"
            st.markdown(html, unsafe_allow_html=True)

        st.markdown('<div class="section-sep">Distribution context</div>', unsafe_allow_html=True)
        d1, d2 = st.columns(2)
        with d1:
            st.plotly_chart(chart_violin_survival(survival_months), use_container_width=True)
        with d2:
            st.plotly_chart(chart_violin_tumor(tumor_size), use_container_width=True)

        st.plotly_chart(chart_nodes_scatter(regional_nodes, tumor_size), use_container_width=True)
# ──────────────────────────────────────────────
# ALL MODEL PREDICTIONS
# ──────────────────────────────────────────────

st.markdown(
    '<div class="section-sep">All Model Predictions</div>',
    unsafe_allow_html=True
)

alive_count = sum(
    1 for r in all_results
    if r["prediction"] == "Alive"
)

dead_count = len(all_results) - alive_count

c1, c2 = st.columns([1,2])

with c1:
    st.plotly_chart(
        chart_consensus_donut(
            alive_count,
            dead_count
        ),
        use_container_width=True
    )

with c2:
    st.plotly_chart(
        chart_alive_gauge_dots(
            all_results
        ),
        use_container_width=True
    )

st.markdown(
    '<div class="section-sep">Model-by-model predictions</div>',
    unsafe_allow_html=True
)

st.plotly_chart(
    chart_model_status_table(
        all_results
    ),
    use_container_width=True
)

tab1, tab2, tab3 = st.tabs(
    [
        "Probability Comparison",
        "Radar",
        "Heatmap"
    ]
)

with tab1:
    st.plotly_chart(
        chart_proba_bars(
            all_results
        ),
        use_container_width=True
    )

with tab2:
    st.plotly_chart(
        chart_radar(
            all_results
        ),
        use_container_width=True
    )

with tab3:
    st.plotly_chart(
        chart_risk_heatmap(
            all_results
        ),
        use_container_width=True
    )
# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — COMPARE ALL MODELS
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📊  Compare Models":
    st.markdown("<h2 style='margin-bottom:0.25rem;'>All-model comparison</h2>", unsafe_allow_html=True)
    st.markdown("<p style='color:#4A6278;font-size:13px;margin-bottom:1.5rem;'>Same patient data, all 8 models — side by side.</p>", unsafe_allow_html=True)

    st.markdown('<div class="section-sep">Patient inputs</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        csm = st.number_input("Survival months", 1, 107, 60, key="c_sm")
        ces = st.selectbox("Estrogen status", ["Positive","Negative"], key="c_es")
    with c2:
        crn = st.number_input("Regional nodes positive", 1, 46, 5, key="c_rn")
        cps = st.selectbox("Progesterone status", ["Positive","Negative"], key="c_ps")
    with c3:
        cts = st.number_input("Tumor size (mm)", 1, 140, 30, key="c_ts")
        cas = st.selectbox("A stage", ["Regional","Distant"], key="c_as")

    run_all = st.button("▶  Run all 8 models", use_container_width=False)

    if run_all:
        input_df = build_input_df(csm, crn, cts, ces, cps, cas)
        scaled   = scaler.transform(input_df)
        results  = run_all_models(scaled)

        alive_count = sum(1 for r in results if r["prediction"]=="Alive")
        dead_count  = len(results) - alive_count

        st.markdown('<div class="section-sep">Consensus</div>', unsafe_allow_html=True)
        con1, con2, con3 = st.columns([1,1,2])
        with con1:
            consensus = "ALIVE" if alive_count >= dead_count else "DEAD"
            color = GREEN if consensus == "ALIVE" else RED
            st.markdown(f"""
            <div class="onco-card" style="text-align:center;border-color:{color}33;">
                <div class="onco-label">Consensus</div>
                <div style="font-size:28px;font-weight:700;color:{color};font-family:'Syne',sans-serif;">{consensus}</div>
                <div style="font-size:12px;color:#4A6278;margin-top:4px;">{alive_count}/8 predict alive</div>
            </div>""", unsafe_allow_html=True)
        with con2:
            st.plotly_chart(chart_consensus_donut(alive_count, dead_count), use_container_width=True)
        with con3:
            st.plotly_chart(chart_alive_gauge_dots(results), use_container_width=True)

        st.markdown('<div class="section-sep">Model-by-model predictions</div>', unsafe_allow_html=True)

        # Results table
        table_html = """<table style="width:100%;border-collapse:collapse;font-size:13px;font-family:'Syne',sans-serif;">
        <thead><tr>
          <th style="text-align:left;padding:8px 12px;font-size:10px;color:#4A6278;font-weight:700;
              text-transform:uppercase;letter-spacing:0.08em;border-bottom:1px solid #1C2A38;">Model</th>
          <th style="padding:8px 12px;font-size:10px;color:#4A6278;font-weight:700;
              text-transform:uppercase;letter-spacing:0.08em;border-bottom:1px solid #1C2A38;text-align:center;">Prediction</th>
          <th style="padding:8px 12px;font-size:10px;color:#4A6278;font-weight:700;
              text-transform:uppercase;letter-spacing:0.08em;border-bottom:1px solid #1C2A38;">Alive probability</th>
          <th style="padding:8px 12px;font-size:10px;color:#4A6278;font-weight:700;
              text-transform:uppercase;letter-spacing:0.08em;border-bottom:1px solid #1C2A38;">Dead probability</th>
        </tr></thead><tbody>"""
        for r in results:
            pred_color = "#4ADB6A" if r["prediction"]=="Alive" else "#FF5555"
            pred_bg    = "rgba(16,60,20,0.5)" if r["prediction"]=="Alive" else "rgba(60,16,16,0.5)"
            mc = MODEL_COLORS.get(r["model"], "#888")
            ap = r["alive_pct"]
            dp = r["dead_pct"]
            ap_bar = f"""<div style="display:flex;align-items:center;gap:8px;">
              <div style="flex:1;background:#0F1922;border-radius:4px;height:6px;overflow:hidden;">
                <div style="width:{ap if ap else 0}%;background:#4ADB6A;height:100%;border-radius:4px;"></div></div>
              <span style="font-size:12px;color:#C8D4E0;min-width:38px;">{f"{ap}%" if ap is not None else "N/A"}</span></div>""" if ap is not None else '<span style="color:#4A6278;font-size:12px;">N/A</span>'
            dp_bar = f"""<div style="display:flex;align-items:center;gap:8px;">
              <div style="flex:1;background:#0F1922;border-radius:4px;height:6px;overflow:hidden;">
                <div style="width:{dp if dp else 0}%;background:#FF5555;height:100%;border-radius:4px;"></div></div>
              <span style="font-size:12px;color:#C8D4E0;min-width:38px;">{f"{dp}%" if dp is not None else "N/A"}</span></div>""" if dp is not None else '<span style="color:#4A6278;font-size:12px;">N/A</span>'
            table_html += f"""<tr style="border-bottom:1px solid #0F1922;">
              <td style="padding:11px 12px;font-weight:600;color:{mc};">{r["model"]}</td>
              <td style="padding:11px 12px;text-align:center;">
                <span style="background:{pred_bg};color:{pred_color};padding:3px 12px;
                  border-radius:99px;font-size:12px;font-weight:600;border:1px solid {pred_color}33;">
                  {r["prediction"]}</span></td>
              <td style="padding:11px 12px;min-width:180px;">{ap_bar}</td>
              <td style="padding:11px 12px;min-width:180px;">{dp_bar}</td>
            </tr>"""
        table_html += "</tbody></table>"
        st.markdown(f'<div class="onco-card" style="padding:0;overflow:hidden;">{table_html}</div>', unsafe_allow_html=True)

        st.markdown('<div class="section-sep">Charts</div>', unsafe_allow_html=True)

        tab0, tab1, tab2, tab3 = st.tabs(["Status table", "Bar comparison", "Radar", "Heatmap"])
        with tab0:
            st.plotly_chart(chart_model_status_table(results), use_container_width=True)
        with tab1:
            st.plotly_chart(chart_proba_bars(results), use_container_width=True)
        with tab2:
            st.plotly_chart(chart_radar(results), use_container_width=True)
        with tab3:
            st.plotly_chart(chart_risk_heatmap(results), use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — DATA EXPLORER
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📈  Data Explorer":
    st.markdown("<h2 style='margin-bottom:0.25rem;'>Population data explorer</h2>", unsafe_allow_html=True)
    st.markdown("<p style='color:#4A6278;font-size:13px;margin-bottom:1.5rem;'>Distribution analysis of survival-related clinical features across the reference cohort.</p>", unsafe_allow_html=True)

    alive_n = len(REF[REF["Status"]=="Alive"])
    dead_n  = len(REF[REF["Status"]=="Dead"])
    e1, e2, e3, e4 = st.columns(4)
    e1.metric("Total patients", len(REF))
    e2.metric("Alive", alive_n)
    e3.metric("Dead", dead_n)
    e4.metric("Survival rate", f"{round(alive_n/len(REF)*100,1)}%")

    st.markdown('<div class="section-sep">Violin plots — feature distributions</div>', unsafe_allow_html=True)
    v1, v2 = st.columns(2)
    with v1:
        st.plotly_chart(chart_violin_survival(60), use_container_width=True)
    with v2:
        st.plotly_chart(chart_violin_tumor(30), use_container_width=True)

    v3, v4 = st.columns(2)
    with v3:
        st.plotly_chart(chart_node_violin(), use_container_width=True)
    with v4:
        # Age violin
        fig = go.Figure()
        for status, color in [("Alive", GREEN), ("Dead", RED)]:
            sub = REF[REF["Status"]==status]["Age"]
            fig.add_trace(go.Violin(
                y=sub, name=status, box_visible=True,
                meanline_visible=True, fillcolor=color.replace("FF","44"),
                line_color=color, opacity=0.9,
            ))
        fig.update_layout(**PLOTLY_LAYOUT,
            title=dict(text="Age distribution — Alive vs Dead", font=dict(size=14, color=TEXT_COLOR)),
            height=320, yaxis=dict(**AXIS_STYLE, title="Age (years)"),
            violingap=0.3, violinmode="overlay",
            legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=TEXT_COLOR)))
        st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="section-sep">Histograms</div>', unsafe_allow_html=True)
    h1, h2 = st.columns(2)
    with h1:
        st.plotly_chart(chart_survival_months_hist(60), use_container_width=True)
    with h2:
        # Nodes histogram
        fig = go.Figure()
        for status, color in [("Alive", GREEN), ("Dead", RED)]:
            sub = REF[REF["Status"]==status]["Regional_Nodes"]
            fig.add_trace(go.Histogram(x=sub, name=status, nbinsx=15,
                                       marker_color=color, opacity=0.6))
        fig.update_layout(**PLOTLY_LAYOUT, barmode="overlay",
            title=dict(text="Regional nodes histogram", font=dict(size=14, color=TEXT_COLOR)),
            height=300, xaxis=dict(**AXIS_STYLE, title="Nodes"),
            yaxis=dict(**AXIS_STYLE, title="Count"),
            legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=TEXT_COLOR)))
        st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="section-sep">Scatter — nodes vs tumor size</div>', unsafe_allow_html=True)
    fig_sc = go.Figure()
    for status, color in [("Alive", GREEN), ("Dead", RED)]:
        sub = REF[REF["Status"]==status]
        fig_sc.add_trace(go.Scatter(
            x=sub["Regional_Nodes"], y=sub["Tumor_Size"],
            mode="markers", name=status,
            marker=dict(size=5, color=color, opacity=0.45, line=dict(width=0)),
        ))
    fig_sc.update_layout(**PLOTLY_LAYOUT,
        title=dict(text="Regional nodes vs tumor size", font=dict(size=14, color=TEXT_COLOR)),
        height=380, xaxis=dict(**AXIS_STYLE, title="Regional nodes"),
        yaxis=dict(**AXIS_STYLE, title="Tumor size (mm)"),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=TEXT_COLOR)))
    st.plotly_chart(fig_sc, use_container_width=True)
