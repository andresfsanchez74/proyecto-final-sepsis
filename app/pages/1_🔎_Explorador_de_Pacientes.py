"""Página 2 · Explorador de pacientes.

Baja del agregado al individuo: la serie de tiempo real de un paciente, y si el modelo
está disponible, su curva de riesgo hora a hora superpuesta al umbral de decisión.
Usa los 50 pacientes versionados en data/sample/ (con .psv reales), no datos sintéticos.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.append(str(Path(__file__).resolve().parent.parent.parent / "src"))
import features  # noqa: E402

from theme import (  # noqa: E402
    COLOR_CONTROL, COLOR_SEPSIS, PALETTE, aviso_modelo_faltante, cargar_manifest_muestra,
    cargar_modelo, encabezado, estilizar, inject_css, leer_paciente_muestra, obtener_cfg,
)

st.set_page_config(page_title="Sepsis UCI · Explorador", page_icon="🔎", layout="wide")
inject_css()

encabezado(
    "Página 2 de 5",
    "Explorador de pacientes",
    "La trayectoria real de un paciente, hora a hora — huecos, fluctuaciones y, si el "
    "modelo está disponible, la curva de riesgo cruzando el umbral de decisión.",
)

cfg = obtener_cfg()
manifest = cargar_manifest_muestra()
paquete, umbral, error_modelo = cargar_modelo()


@st.cache_data(show_spinner=False)
def _estado_deteccion(_paquete, umbral):
    """Para cada paciente de la muestra: ¿el modelo lo alarma antes del onset real?"""
    import serving
    filas = []
    for _, r in manifest.iterrows():
        df_pac = leer_paciente_muestra(r["pid"], r["hosp"])
        pred = serving.predecir_secuencia(df_pac, cfg, umbral=umbral)
        onset = df_pac.loc[df_pac["SepsisLabel"] == 1, "ICULOS"].min()
        detectado = False
        if pd.notna(onset):
            previas = pred.loc[pred["ICULOS"] <= onset, "alerta"]
            detectado = bool(previas.any())
        filas.append({"pid": r["pid"], "hosp": r["hosp"], "septico": r["septico"], "detectado": detectado})
    return pd.DataFrame(filas)


col_sel, col_btn = st.columns([2.2, 1.6])
with col_sel:
    etiquetas = manifest.apply(lambda r: f"{r['pid']} · Hospital {r['hosp']} · {'séptico' if r['septico'] else 'sin sepsis'}", axis=1)
    idx_map = dict(zip(etiquetas, manifest["pid"]))
    if "pid_explorador" not in st.session_state:
        st.session_state["pid_explorador"] = manifest["pid"].iloc[0]
    default_label = next((e for e, p in idx_map.items() if p == st.session_state["pid_explorador"]), etiquetas.iloc[0])
    elegido = st.selectbox("Paciente", options=list(etiquetas), index=list(etiquetas).index(default_label))
    st.session_state["pid_explorador"] = idx_map[elegido]

with col_btn:
    st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
    b1, b2, b3 = st.columns(3)
    if paquete is not None:
        estado = _estado_deteccion(paquete, umbral)
        if b1.button("✓ Detectado", width="stretch"):
            cand = estado[estado["detectado"]]
            if len(cand):
                st.session_state["pid_explorador"] = cand.sample(1)["pid"].iloc[0]
                st.rerun()
        if b2.button("✗ No detectado", width="stretch"):
            cand = estado[estado["septico"] == 1][~estado["detectado"]]
            if len(cand):
                st.session_state["pid_explorador"] = cand.sample(1)["pid"].iloc[0]
                st.rerun()
    if b3.button("🎲 Aleatorio", width="stretch"):
        st.session_state["pid_explorador"] = manifest.sample(1)["pid"].iloc[0]
        st.rerun()

pid = st.session_state["pid_explorador"]
fila_manifest = manifest.loc[manifest["pid"] == pid].iloc[0]
df_pac = leer_paciente_muestra(pid, fila_manifest["hosp"])
onset = df_pac.loc[df_pac["SepsisLabel"] == 1, "ICULOS"].min()

# ---------------------------------------------------------------------------
# Encabezado del paciente
# ---------------------------------------------------------------------------
c1, c2, c3, c4 = st.columns(4)
c1.metric("Paciente", pid)
c2.metric("Hospital", fila_manifest["hosp"])
c3.metric("Horas en UCI", f"{df_pac['ICULOS'].max():.0f}")
c4.metric("Inicio de sepsis", f"hora {onset:.0f}" if pd.notna(onset) else "no desarrolla")

st.markdown("<br>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Series de tiempo + curva de riesgo
# ---------------------------------------------------------------------------
st.markdown('<div class="card"><div class="card-title">Signos vitales, hora a hora</div>', unsafe_allow_html=True)
opciones_vitales = [v for v in features.VITALES if v in df_pac.columns]
default_vars = [v for v in ["HR", "Temp", "Resp", "MAP"] if v in opciones_vitales]
seleccion = st.multiselect("Variables a graficar", opciones_vitales, default=default_vars)

if seleccion:
    fig = go.Figure()
    colores = [PALETTE["orange"], PALETTE["slate"], PALETTE["orange_deep"], PALETTE["muted"]]
    for i, var in enumerate(seleccion):
        fig.add_trace(go.Scatter(
            x=df_pac["ICULOS"], y=df_pac[var], mode="lines+markers", name=var,
            line=dict(color=colores[i % len(colores)], width=2), marker=dict(size=4),
            connectgaps=True,
        ))
    if pd.notna(onset):
        fig.add_vline(x=onset, line_dash="dash", line_color=COLOR_SEPSIS, line_width=1.5,
                      annotation_text="Inicio de sepsis", annotation_position="top")
    fig.update_xaxes(title="Horas desde el ingreso a UCI (ICULOS)")
    fig.update_yaxes(title="Valor medido")
    st.plotly_chart(estilizar(fig, height=380), width="stretch")
st.markdown("</div>", unsafe_allow_html=True)

if paquete is not None:
    st.markdown('<div class="card"><div class="card-title">Curva de riesgo del modelo campeón (CatBoost)</div>', unsafe_allow_html=True)
    import serving
    pred = serving.predecir_secuencia(df_pac, cfg, umbral=umbral)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=pred["ICULOS"], y=pred["proba_sepsis"], mode="lines", name="Riesgo estimado",
        line=dict(color=COLOR_SEPSIS, width=2.5), fill="tozeroy",
        fillcolor="rgba(220,107,24,0.12)",
    ))
    fig.add_hline(y=umbral, line_dash="dot", line_color=PALETTE["ink_soft"], line_width=1.3,
                  annotation_text=f"Umbral de alarma ({umbral:.3f})", annotation_position="bottom right")
    if pd.notna(onset):
        fig.add_vline(x=onset, line_dash="dash", line_color=PALETTE["ink"], line_width=1.5,
                      annotation_text="Inicio real", annotation_position="top left")
        cruces = pred.loc[(pred["ICULOS"] <= onset) & pred["alerta"], "ICULOS"]
        if len(cruces):
            primera = cruces.min()
            fig.add_vrect(x0=primera, x1=onset, fillcolor=PALETTE["green"], opacity=0.08, line_width=0,
                          annotation_text=f"{onset - primera:.0f} h de anticipación", annotation_position="top")
    fig.update_xaxes(title="Horas desde el ingreso a UCI (ICULOS)")
    fig.update_yaxes(title="Probabilidad de sepsis", range=[0, 1])
    st.plotly_chart(estilizar(fig, height=340, legend=False), width="stretch")
    st.markdown("</div>", unsafe_allow_html=True)
else:
    aviso_modelo_faltante(error_modelo)

# ---------------------------------------------------------------------------
# Mapa de disponibilidad + descarga
# ---------------------------------------------------------------------------
col5, col6 = st.columns([1.6, 1])

with col5:
    st.markdown('<div class="card"><div class="card-title">Qué se midió y qué no, hora por hora</div>', unsafe_allow_html=True)
    clinicas = [c for c in features.VITALES + features.LABORATORIOS if c in df_pac.columns]
    matriz = df_pac.set_index("ICULOS")[clinicas].notna().astype(int).T
    fig = go.Figure(go.Heatmap(
        z=matriz.values, x=matriz.columns, y=matriz.index,
        colorscale=[[0, PALETTE["beige"]], [1, PALETTE["orange_deep"]]],
        showscale=False, xgap=1, ygap=1,
    ))
    fig.update_xaxes(title="Horas desde el ingreso a UCI")
    fig.update_yaxes(title="", autorange="reversed")
    st.plotly_chart(estilizar(fig, height=460, legend=False), width="stretch")
    st.markdown("</div>", unsafe_allow_html=True)

with col6:
    st.markdown('<div class="card"><div class="card-title">Historia horaria completa</div>', unsafe_allow_html=True)
    st.dataframe(df_pac.drop(columns=["pid", "hosp"]), height=420, width="stretch")
    st.download_button(
        "Descargar CSV", df_pac.to_csv(index=False).encode("utf-8"),
        file_name=f"{pid}_historia.csv", mime="text/csv", width="stretch",
    )
    st.markdown("</div>", unsafe_allow_html=True)
