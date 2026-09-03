"""Página 3 · Desempeño y umbral.

El corazón analítico del dashboard: mover el presupuesto de alarma y ver el trade-off
completo (sensibilidad, falsas alarmas, utility) reaccionar en vivo sobre las 310.997
horas del test — en vez de leerlo como una tabla fija en una diapositiva.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve
import streamlit as st

sys.path.append(str(Path(__file__).resolve().parent.parent.parent / "src"))
import evaluacion  # noqa: E402

from theme import (  # noqa: E402
    COLOR_CONTROL, COLOR_SEPSIS, MODELOS_TABULARES, PALETTE, cargar_test_scored,
    encabezado, estilizar, inject_css, md, obtener_cfg,
)

st.set_page_config(page_title="Sepsis UCI · Desempeño", page_icon="🎯", layout="wide")
inject_css()

encabezado(
    "Página 3 de 5",
    "Desempeño y umbral de decisión",
    "El presupuesto de alarma no es un detalle técnico: es una decisión operativa. "
    "Mueve el control y observa cómo cambian la sensibilidad, las falsas alarmas y el "
    "utility score — la métrica oficial del challenge.",
)

cfg = obtener_cfg()
df = cargar_test_scored()

# ---------------------------------------------------------------------------
# Controles
# ---------------------------------------------------------------------------
ctrl1, ctrl2, ctrl3 = st.columns([2, 1.3, 1.3])
with ctrl1:
    tasa_pct = st.slider("Presupuesto de alarma (% de horas)", 1.0, 10.0, 5.0, step=0.5,
                          help="Porcentaje de horas del test en las que el modelo puede alarmar. "
                               "5% es el valor elegido en el notebook 07 como presupuesto clínicamente tolerable.")
with ctrl2:
    modelo = st.selectbox("Modelo", MODELOS_TABULARES, index=0)
with ctrl3:
    particion = st.radio("Vista", ["Test completo", "Por hospital"], horizontal=True)

tasa = tasa_pct / 100
col_score = f"proba_{modelo}"


@st.cache_data(show_spinner="Recalculando sobre las 310.997 horas del test…")
def _evaluar(modelo_col, tasa, hosp_filtro):
    cfg_local = {**cfg, "evaluacion": {**cfg["evaluacion"], "tasa_alarma_objetivo": tasa}}
    d = df if hosp_filtro is None else df[df["hosp"] == hosp_filtro]
    return evaluacion.evaluar(d, d[modelo_col], cfg_local, nombre=modelo_col)


@st.cache_data(show_spinner="Trazando la curva de utility completa (una sola vez por modelo)…")
def _barrido(modelo_col):
    return evaluacion.barrido_de_umbral(df, df[modelo_col], cfg, n_umbrales=18)


res = _evaluar(col_score, tasa, None)
umbral = res["umbral"]
sens_key = [k for k in res if k.startswith("sensibilidad_a_")][0]

# ---------------------------------------------------------------------------
# KPIs que reaccionan al slider
# ---------------------------------------------------------------------------
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Umbral resultante", f"{umbral:.3f}")
c2.metric("Sensibilidad (horas)", f"{res[sens_key]*100:.1f} %")
c3.metric("Utility normalizado", f"{res['utility_normalizado']:.4f}")
c4.metric("Pacientes detectados", f"{res['detectados']} / {res['detectados'] + res['no_detectados_antes_del_evento']}")
c5.metric("Anticipación mediana", f"{res['anticipacion_mediana_h']:.0f} h" if pd.notna(res["anticipacion_mediana_h"]) else "—")

n_alarmas = int(round(tasa * len(df)))
n_aciertos = int(round(res[sens_key] * df["SepsisLabel"].sum()))
md(
    f"""<div class="note">Con este presupuesto, el modelo genera <strong>{n_alarmas:,} alarmas</strong>
    sobre {len(df):,} horas de test. De ellas, aproximadamente <strong>{n_aciertos:,} son sepsis real</strong>
    y el resto —{n_alarmas - n_aciertos:,}— son falsas alarmas: cerca de
    <strong>{(n_alarmas - n_aciertos) / max(n_aciertos, 1):.1f} descartes por acierto</strong>.</div>""",
    unsafe_allow_html=True,
)
st.markdown("<br>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Fila: matriz de confusión + curva de utility
# ---------------------------------------------------------------------------
col1, col2 = st.columns([1, 1.3])

with col1:
    st.markdown('<div class="card"><div class="card-title">Matriz de confusión al umbral actual</div>', unsafe_allow_html=True)
    pred = (df[col_score] >= umbral).astype(int)
    cm = confusion_matrix(df["SepsisLabel"], pred)
    etiquetas_cm = ["No sepsis", "Sepsis"]
    fig = go.Figure(go.Heatmap(
        z=cm, x=["Pred: No", "Pred: Sí"], y=["Real: No", "Real: Sí"],
        text=[[f"{v:,}" for v in fila] for fila in cm], texttemplate="%{text}",
        textfont=dict(size=16, family="IBM Plex Mono, monospace"),
        colorscale=[[0, PALETTE["paper"]], [1, PALETTE["orange"]]], showscale=False,
    ))
    fig.update_yaxes(autorange="reversed")
    st.plotly_chart(estilizar(fig, height=300, legend=False), width="stretch")
    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    st.markdown('<div class="card"><div class="card-title">Utility vs. presupuesto de alarma</div>', unsafe_allow_html=True)
    curva = _barrido(col_score)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=curva["tasa_alarma"] * 100, y=curva["utility"], mode="lines",
                              line=dict(color=COLOR_SEPSIS, width=2.5)))
    fig.add_vline(x=tasa_pct, line_dash="dash", line_color=PALETTE["ink"], line_width=1.5,
                  annotation_text="Presupuesto actual", annotation_position="top")
    fig.update_xaxes(title="Presupuesto de alarma (%)")
    fig.update_yaxes(title="Utility normalizado")
    st.plotly_chart(estilizar(fig, height=300, legend=False), width="stretch")
    st.markdown("</div>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Fila: ROC / PR + distribución de probabilidades
# ---------------------------------------------------------------------------
col3, col4 = st.columns(2)

with col3:
    st.markdown('<div class="card"><div class="card-title">Curvas ROC y Precision-Recall</div>', unsafe_allow_html=True)
    fpr, tpr, _ = roc_curve(df["SepsisLabel"], df[col_score])
    prec, rec, _ = precision_recall_curve(df["SepsisLabel"], df[col_score])
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"ROC (AUROC {res['AUROC']:.3f})",
                              line=dict(color=COLOR_SEPSIS, width=2.5)))
    fig.add_trace(go.Scatter(x=rec, y=prec, mode="lines", name=f"PR (AUPRC {res['AUPRC']:.3f})",
                              line=dict(color=COLOR_CONTROL, width=2.5), xaxis="x2", yaxis="y2"))
    fig.add_shape(type="line", x0=0, y0=0, x1=1, y1=1, line=dict(color=PALETTE["rule"], dash="dot"))
    fig.update_layout(
        xaxis=dict(domain=[0, 0.46], title="Tasa falsos positivos"),
        yaxis=dict(title="Sensibilidad (ROC)"),
        xaxis2=dict(domain=[0.54, 1], title="Recall", anchor="y2"),
        yaxis2=dict(title="Precisión", anchor="x2"),
    )
    st.plotly_chart(estilizar(fig, height=340), width="stretch")
    st.markdown("</div>", unsafe_allow_html=True)

with col4:
    st.markdown('<div class="card"><div class="card-title">Distribución de probabilidades predichas</div>', unsafe_allow_html=True)
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=df.loc[df["SepsisLabel"] == 0, col_score], name="No sepsis",
                                histnorm="probability density", marker_color=COLOR_CONTROL, opacity=0.6, nbinsx=60))
    fig.add_trace(go.Histogram(x=df.loc[df["SepsisLabel"] == 1, col_score], name="Sepsis",
                                histnorm="probability density", marker_color=COLOR_SEPSIS, opacity=0.6, nbinsx=60))
    fig.add_vline(x=umbral, line_dash="dash", line_color=PALETTE["ink"], line_width=1.5,
                  annotation_text="Umbral", annotation_position="top")
    fig.update_layout(barmode="overlay")
    fig.update_xaxes(title="Probabilidad predicha")
    fig.update_yaxes(title="Densidad")
    st.plotly_chart(estilizar(fig, height=340), width="stretch")
    st.markdown("</div>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Vista partida por hospital
# ---------------------------------------------------------------------------
if particion == "Por hospital":
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="card"><div class="card-title">El mismo umbral global, aplicado por separado en cada hospital</div>', unsafe_allow_html=True)

    res_a = _evaluar(col_score, tasa, "A")
    res_b = _evaluar(col_score, tasa, "B")
    tabla = pd.DataFrame([
        {"Hospital": "A", "Utility": res_a["utility_normalizado"], "AUROC": res_a["AUROC"],
         "Sensibilidad": res_a[sens_key], "Anticipación mediana (h)": res_a["anticipacion_mediana_h"]},
        {"Hospital": "B", "Utility": res_b["utility_normalizado"], "AUROC": res_b["AUROC"],
         "Sensibilidad": res_b[sens_key], "Anticipación mediana (h)": res_b["anticipacion_mediana_h"]},
    ])

    colh1, colh2 = st.columns([1, 1.4])
    with colh1:
        st.dataframe(tabla.set_index("Hospital").round(4), width="stretch")
    with colh2:
        fig = go.Figure()
        fig.add_trace(go.Bar(x=["Utility", "Sensibilidad"], y=[res_a["utility_normalizado"], res_a[sens_key]],
                              name="Hospital A", marker_color=COLOR_SEPSIS))
        fig.add_trace(go.Bar(x=["Utility", "Sensibilidad"], y=[res_b["utility_normalizado"], res_b[sens_key]],
                              name="Hospital B", marker_color=COLOR_CONTROL))
        fig.update_layout(barmode="group")
        st.plotly_chart(estilizar(fig, height=280), width="stretch")
    md(
        """<div class="note">El mismo umbral alarma en proporciones distintas en cada hospital
        (el 5% se calibra sobre el test mezclado). "Mejor" depende de qué se mida: como
        clasificador, mejor en B; como sistema de alarma ya desplegado, mejor en A.</div>""",
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)
