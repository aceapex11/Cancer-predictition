import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go

# ── PAGE CONFIG ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CancerPredict · Survival Model",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── GLOBAL CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── fonts & base ── */
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

/* ── sidebar ── */
[data-testid="stSidebar"] {
    background: #FFFFFF;
    border-right: 1px solid #EBEBEB;
    padding-top: 1.5rem;
}
[data-testid="stSidebar"] .block-container { padding: 0 1rem; }

/* hide default streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }
[data-testid="stDecoration"] { display: none; }

/* ── page background ── */
.main .block-container {
    padding: 2rem 2.5rem;
    max-width: 1100px;
}

/* ── metric cards ── */
[data-testid="metric-container"] {
    background: #F9F9F8;
    border: 1px solid #EBEBEB;
    border-radius: 10px;
    padding: 1rem 1.25rem;
}

/* ── buttons ── */
.stButton > button {
    background: #D85A30 !important;
    color: #fff !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 500 !important;
    font-size: 14px !important;
    padding: 0.55rem 1.4rem !important;
    transition: background 0.15s !important;
}
.stButton > button:hover {
    background: #993C1D !important;
}

/* ── selectbox & number input ── */
[data-testid="stSelectbox"] > div > div,
[data-testid="stNumberInput"] input {
    border-radius: 8px !important;
    border: 1px solid #DDDDD8 !important;
    background: #F9F9F8 !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 13px !important;
}

/* ── section label ── */
.section-label {
    font-size: 11px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.07em;
    color: #888;
    margin-bottom: 0.6rem;
    margin-top: 0.2rem;
}

/* ── result banners ── */
.banner-alive {
    background: #EAF3DE;
    border: 1px solid #C0DD97;
    border-radius: 10px;
    padding: 14px 18px;
    display: flex;
    align-items: center;
    gap: 10px;
    font-size: 15px;
    font-weight: 500;
    color: #27500A;
    margin-bottom: 1rem;
}
.banner-dead {
    background: #FCEBEB;
    border: 1px solid #F7C1C1;
    border-radius: 10px;
    padding: 14px 18px;
    display: flex;
    align-items: center;
    gap: 10px;
    font-size: 15px;
    font-weight: 500;
    color: #791F1F;
    margin-bottom: 1rem;
}

/* ── patient table ── */
.patient-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 13px;
    margin-top: 0.5rem;
}
.patient-table td {
    padding: 6px 4px;
    border-bottom: 1px solid #F0F0EC;
}
.patient-table td:first-child { color: #888; }
.patient-table td:last-child { font-weight: 500; text-align: right; }

/* ── compare badge ── */
.badge-alive {
    background: #EAF3DE; color: #27500A;
    padding: 3px 10px; border-radius: 99px;
    font-size: 12px; font-weight: 500;
    display: inline-block;
}
.badge-dead {
    background: #FCEBEB; color: #791F1F;
    padding: 3px 10px; border-radius: 99px;
    font-size: 12px; font-weight: 500;
    display: inline-block;
}

/* ── consensus card ── */
.consensus-card {
    background: #F9F9F8;
    border: 1px solid #EBEBEB;
    border-radius: 10px;
    padding: 1.1rem 1.4rem;
    margin-top: 1rem;
}
.consensus-num {
    font-size: 30px;
    font-weight: 600;
    line-height: 1;
}
.consensus-label {
    font-size: 11px;
    color: #888;
    margin-top: 2px;
}
</style>
""", unsafe_allow_html=True)

# ── LOAD MODEL ─────────────────────────────────────────────────────────────────
@st.cache_resource
def load_bundle():
    return joblib.load("breast_cancer_models_compressed.pkl")

bundle = load_bundle()
scaler = bundle["scaler"]

MODEL_NAMES = [
    "Logistic Regression", "Decision Tree", "Random Forest",
    "KNN", "SVM", "Naive Bayes", "Gradient Boosting", "XGBoost"
]

# ── SIDEBAR ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="display:flex;align-items:center;gap:10px;padding:0 0.25rem 1.2rem;border-bottom:1px solid #EBEBEB;margin-bottom:1rem;">
        <div style="width:32px;height:32px;border-radius:8px;background:#F5C4B3;display:flex;align-items:center;justify-content:center;font-size:16px;">🩺</div>
        <div>
            <div style="font-weight:600;font-size:14px;line-height:1.3;">CancerPredict</div>
            <div style="font-size:11px;color:#888;">Survival model</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    page = st.radio(
        "Navigation",
        ["Prediction", "Compare Models"],
        label_visibility="collapsed"
    )

# ── SHARED INPUT WIDGET ────────────────────────────────────────────────────────
def render_inputs(prefix=""):
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="section-label">Clinical data</div>', unsafe_allow_html=True)
        survival_months = st.number_input("Survival months", 1, 107, 60, key=f"{prefix}months")
        regional_nodes  = st.number_input("Regional nodes positive", 1, 46, 5, key=f"{prefix}nodes")
        tumor_size      = st.number_input("Tumor size (mm)", 1, 140, 30, key=f"{prefix}tumor")
    with col2:
        st.markdown('<div class="section-label">Receptor & stage status</div>', unsafe_allow_html=True)
        estrogen_status      = st.selectbox("Estrogen status",      ["Positive", "Negative"], key=f"{prefix}estrogen")
        progesterone_status  = st.selectbox("Progesterone status",  ["Positive", "Negative"], key=f"{prefix}prog")
        a_stage              = st.selectbox("A stage",              ["Regional",  "Distant"],  key=f"{prefix}astage")
    return survival_months, regional_nodes, tumor_size, estrogen_status, progesterone_status, a_stage


def build_input_df(survival_months, regional_nodes, tumor_size,
                   estrogen_status, progesterone_status, a_stage):
    return pd.DataFrame([[
        survival_months,
        regional_nodes,
        0 if estrogen_status == "Positive" else 1,
        0 if progesterone_status == "Positive" else 1,
        tumor_size,
        0 if a_stage == "Regional" else 1,
    ]], columns=[
        "Survival Months", "Reginol Node Positive",
        "Estrogen Status", "Progesterone Status",
        "Tumor Size", "A Stage",
    ])


# ── PREDICTION PAGE ────────────────────────────────────────────────────────────
if page == "Prediction":
    st.markdown("## Survival prediction")
    st.markdown(
        "<p style='color:#888;font-size:14px;margin-top:-8px;margin-bottom:1.5rem;'>"
        "Enter patient data and select a model to predict survival status.</p>",
        unsafe_allow_html=True
    )

    # Model chips via pills-style selectbox
    st.markdown('<div class="section-label">Select model</div>', unsafe_allow_html=True)
    model_name = st.selectbox("Model", MODEL_NAMES, label_visibility="collapsed")
    st.markdown("<br>", unsafe_allow_html=True)

    survival_months, regional_nodes, tumor_size, \
        estrogen_status, progesterone_status, a_stage = render_inputs("pred_")

    st.markdown("<br>", unsafe_allow_html=True)
    run = st.button("⚡  Run prediction", use_container_width=False)

    if run:
        input_df    = build_input_df(survival_months, regional_nodes, tumor_size,
                                     estrogen_status, progesterone_status, a_stage)
        scaled      = scaler.transform(input_df)
        model       = bundle[model_name]
        prediction  = model.predict(scaled)[0]
        probs       = model.predict_proba(scaled)[0]
        alive_pct   = round(probs[0] * 100, 1)
        dead_pct    = round(probs[1] * 100, 1)

        st.markdown("---")
        st.markdown("#### Result")

        if prediction == 0:
            st.markdown(
                '<div class="banner-alive">✅ &nbsp; Predicted status: <strong>Alive</strong></div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                '<div class="banner-dead">❌ &nbsp; Predicted status: <strong>Dead</strong></div>',
                unsafe_allow_html=True
            )

        # Probability metrics
        m1, m2 = st.columns(2)
        m1.metric("Alive probability", f"{alive_pct}%")
        m2.metric("Dead probability",  f"{dead_pct}%")

        # Probability bar chart
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=["Alive", "Dead"],
            y=[alive_pct, dead_pct],
            marker_color=["#97C459", "#E24B4A"],
            text=[f"{alive_pct}%", f"{dead_pct}%"],
            textposition="outside",
            width=0.4,
        ))
        fig.update_layout(
            height=260,
            margin=dict(t=20, b=10, l=0, r=0),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            showlegend=False,
            yaxis=dict(range=[0, 115], showgrid=False, showticklabels=False),
            xaxis=dict(showgrid=False),
            font=dict(family="DM Sans", size=13),
        )
        st.plotly_chart(fig, use_container_width=True)

        # Patient summary table
        st.markdown('<div class="section-label" style="margin-top:0.5rem;">Patient summary</div>',
                    unsafe_allow_html=True)
        rows = [
            ("Model",             model_name),
            ("Survival months",   survival_months),
            ("Regional nodes",    regional_nodes),
            ("Tumor size",        f"{tumor_size} mm"),
            ("Estrogen status",   estrogen_status),
            ("Progesterone",      progesterone_status),
            ("A stage",           a_stage),
        ]
        table_html = '<table class="patient-table">'
        for k, v in rows:
            table_html += f"<tr><td>{k}</td><td>{v}</td></tr>"
        table_html += "</table>"
        st.markdown(table_html, unsafe_allow_html=True)


# ── COMPARE MODELS PAGE ────────────────────────────────────────────────────────
elif page == "Compare Models":
    st.markdown("## Compare all models")
    st.markdown(
        "<p style='color:#888;font-size:14px;margin-top:-8px;margin-bottom:1.5rem;'>"
        "Run the same patient data through all 8 models simultaneously.</p>",
        unsafe_allow_html=True
    )

    survival_months, regional_nodes, tumor_size, \
        estrogen_status, progesterone_status, a_stage = render_inputs("cmp_")

    st.markdown("<br>", unsafe_allow_html=True)
    run_all = st.button("▶  Run all models", use_container_width=False)

    if run_all:
        input_df = build_input_df(survival_months, regional_nodes, tumor_size,
                                  estrogen_status, progesterone_status, a_stage)
        scaled = scaler.transform(input_df)

        results = []
        for name in MODEL_NAMES:
            m    = bundle[name]
            pred = m.predict(scaled)[0]
            prob = m.predict_proba(scaled)[0]
            results.append({
                "model":     name,
                "prediction": "Alive" if pred == 0 else "Dead",
                "alive_pct": round(prob[0] * 100, 1),
            })

        st.markdown("---")
        st.markdown("#### Model predictions")

        # Render table
        table_html = """
        <table style="width:100%;border-collapse:collapse;font-size:13px;">
          <thead>
            <tr>
              <th style="text-align:left;padding:8px 10px;font-size:11px;color:#888;
                         font-weight:600;text-transform:uppercase;letter-spacing:0.06em;
                         border-bottom:1px solid #EBEBEB;">Model</th>
              <th style="text-align:left;padding:8px 10px;font-size:11px;color:#888;
                         font-weight:600;text-transform:uppercase;letter-spacing:0.06em;
                         border-bottom:1px solid #EBEBEB;">Prediction</th>
              <th style="text-align:left;padding:8px 10px;font-size:11px;color:#888;
                         font-weight:600;text-transform:uppercase;letter-spacing:0.06em;
                         border-bottom:1px solid #EBEBEB;">Alive probability</th>
            </tr>
          </thead>
          <tbody>
        """
        for r in results:
            badge = (
                f'<span class="badge-alive">✅ Alive</span>'
                if r["prediction"] == "Alive"
                else f'<span class="badge-dead">❌ Dead</span>'
            )
            bar_color = "#97C459" if r["prediction"] == "Alive" else "#E24B4A"
            bar_html = (
                f'<div style="display:flex;align-items:center;gap:8px;">'
                f'<div style="flex:1;background:#F0F0EC;border-radius:99px;height:7px;overflow:hidden;">'
                f'<div style="width:{r["alive_pct"]}%;background:{bar_color};height:100%;border-radius:99px;"></div>'
                f'</div>'
                f'<span style="font-size:12px;color:#555;min-width:36px;">{r["alive_pct"]}%</span>'
                f'</div>'
            )
            table_html += (
                f'<tr style="border-bottom:1px solid #F5F5F2;">'
                f'<td style="padding:10px 10px;font-weight:500;">{r["model"]}</td>'
                f'<td style="padding:10px 10px;">{badge}</td>'
                f'<td style="padding:10px 10px;min-width:160px;">{bar_html}</td>'
                f'</tr>'
            )
        table_html += "</tbody></table>"
        st.markdown(table_html, unsafe_allow_html=True)

        # Consensus summary
        alive_count = sum(1 for r in results if r["prediction"] == "Alive")
        dead_count  = len(results) - alive_count
        consensus   = "Alive" if alive_count >= dead_count else "Dead"
        cons_color  = "#27500A" if consensus == "Alive" else "#791F1F"
        pct         = round(alive_count / len(results) * 100)

        bar_fill_color = "#97C459" if consensus == "Alive" else "#E24B4A"

        st.markdown(f"""
        <div class="consensus-card" style="margin-top:1.5rem;">
            <div style="display:flex;align-items:center;gap:1.5rem;">
                <div style="flex-shrink:0;text-align:center;">
                    <div class="consensus-num" style="color:{cons_color};">{alive_count}/8</div>
                    <div class="consensus-label">models agree</div>
                </div>
                <div style="flex:1;">
                    <div style="font-size:13px;margin-bottom:8px;">
                        Consensus: <strong style="color:{cons_color};">{consensus}</strong>
                    </div>
                    <div style="background:#EBEBEB;border-radius:99px;height:9px;overflow:hidden;">
                        <div style="width:{pct}%;background:{bar_fill_color};
                                    height:100%;border-radius:99px;transition:width 0.4s;"></div>
                    </div>
                    <div style="display:flex;justify-content:space-between;
                                font-size:11px;color:#888;margin-top:5px;">
                        <span>{alive_count} predict alive</span>
                        <span>{dead_count} predict dead</span>
                    </div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
