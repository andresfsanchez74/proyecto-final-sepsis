"""Página 5 · Interpretabilidad y límites.

Cierra el dashboard con la honestidad que es la marca del proyecto: en qué se fija el
modelo, qué tan sobre-representado está el contexto administrativo, y qué tan predecible
es la sepsis en principio con los datos disponibles.
"""

import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.append(str(Path(__file__).resolve().parent.parent.parent / "src"))

from theme import (  # noqa: E402
    COLOR_CONTROL, COLOR_SEPSIS, PALETTE, cargar_importancia_shap, cargar_test_scored,
    encabezado, estilizar, inject_css, md, obtener_cfg, ruta_figura,
)

st.set_page_config(page_title="Sepsis UCI · Interpretabilidad", page_icon="🧬", layout="wide")
inject_css()

encabezado(
    "Página 5 de 5",
    "Interpretabilidad y límites",
    "El modelo encuentra señal fisiológica real — pero también se apoya en contexto "
    "administrativo que no viaja a otro hospital. Esta página muestra la evidencia, no "
    "solo la conclusión.",
)

cfg = obtener_cfg()
shap_df = cargar_importancia_shap()

# ---------------------------------------------------------------------------
# Top 15 + peso por familia
# ---------------------------------------------------------------------------
col1, col2 = st.columns([1.3, 1])

FAMILIA_COLOR = {
    "Estado fisiológico": COLOR_SEPSIS,
    "Práctica de medición": PALETTE["muted"],
    "Contexto administrativo": COLOR_CONTROL,
}

with col1:
    st.markdown('<div class="card"><div class="card-title">Las 15 variables que más pesan (SHAP medio absoluto)</div>', unsafe_allow_html=True)
    top15 = shap_df.sort_values("shap_medio_abs", ascending=True).tail(15)
    colores = [FAMILIA_COLOR.get(f, PALETTE["muted"]) for f in top15["familia"]]
    fig = go.Figure(go.Bar(
        x=top15["shap_medio_abs"], y=top15["feature"], orientation="h", marker_color=colores,
    ))
    fig.update_xaxes(title="|SHAP| medio")
    fig.update_yaxes(title="")
    st.plotly_chart(estilizar(fig, height=460, legend=False), width="stretch")
    leyenda = " · ".join(f'<span style="color:{c}">■</span> {f}' for f, c in FAMILIA_COLOR.items())
    st.markdown(f"<p style='font-size:13px;color:#8B7D6B'>{leyenda}</p>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    st.markdown('<div class="card"><div class="card-title">¿El peso es proporcional al número de columnas?</div>', unsafe_allow_html=True)
    resumen = shap_df.groupby("familia").agg(n_cols=("feature", "size"), peso=("shap_medio_abs", "sum"))
    resumen["pct_columnas"] = resumen["n_cols"] / resumen["n_cols"].sum() * 100
    resumen["pct_peso"] = resumen["peso"] / resumen["peso"].sum() * 100
    resumen = resumen.reset_index()

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, 100], y=[0, 100], mode="lines", line=dict(color=PALETTE["rule"], dash="dot"),
                              showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(
        x=resumen["pct_columnas"], y=resumen["pct_peso"], mode="markers+text",
        marker=dict(size=resumen["n_cols"] / resumen["n_cols"].max() * 40 + 14,
                    color=[FAMILIA_COLOR.get(f, PALETTE["muted"]) for f in resumen["familia"]]),
        text=resumen["familia"].str.split().str[0], textposition="top center",
        showlegend=False,
    ))
    fig.update_xaxes(title="% de las columnas del modelo")
    fig.update_yaxes(title="% del peso SHAP total")
    st.plotly_chart(estilizar(fig, height=460, legend=False), width="stretch")
    md(
        """<p style='font-size:13.5px;color:#564B3F'>Por encima de la diagonal = sobre-representado.
        El contexto administrativo concentra más peso del que le tocaría por su número de columnas;
        la práctica de medición, menos.</p>""",
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Simulador de impacto operativo
# ---------------------------------------------------------------------------
st.markdown('<div class="card"><div class="card-title">Traductor: ¿cuántas alarmas por turno de enfermería?</div>', unsafe_allow_html=True)
ci1, ci2, ci3 = st.columns(3)
camas = ci1.number_input("Camas en la UCI", 4, 60, 20)
horas_turno = ci2.number_input("Horas por turno", 4, 24, 12)
tasa_alarma_pct = ci3.slider("Presupuesto de alarma asumido (%)", 1.0, 10.0, 5.0, step=0.5, key="tasa_pagina5")

df_test = cargar_test_scored()
tasa_alarma = tasa_alarma_pct / 100
prevalencia = df_test["SepsisLabel"].mean()

import evaluacion


@st.cache_data(show_spinner="Calculando el punto de operación…")
def _evaluar_campeon(tasa):
    cfg_local = {**cfg, "evaluacion": {**cfg["evaluacion"], "tasa_alarma_objetivo": tasa}}
    return evaluacion.evaluar(df_test, df_test["proba_CatBoost"], cfg_local, nombre="CatBoost")


res5 = _evaluar_campeon(tasa_alarma)
n_alarmas_test = tasa_alarma * len(df_test)
n_aciertos_test = res5[[k for k in res5 if k.startswith("sensibilidad_a_")][0]] * df_test["SepsisLabel"].sum()
precision_operativa = n_aciertos_test / n_alarmas_test if n_alarmas_test else 0

horas_cama_turno = camas * horas_turno
alarmas_turno = tasa_alarma * horas_cama_turno
aciertos_turno = alarmas_turno * precision_operativa

r1, r2, r3 = st.columns(3)
r1.metric("Alarmas esperadas por turno", f"{alarmas_turno:.1f}")
r2.metric("De ellas, sepsis real", f"{aciertos_turno:.1f}")
r3.metric("1 acierto real cada…", f"{alarmas_turno/max(aciertos_turno,0.01):.1f} alarmas")

md(
    f"""<p style='font-size:14px;color:#564B3F'>En una UCI de <strong>{camas} camas</strong> con
    turnos de <strong>{horas_turno} h</strong>, al {tasa_alarma_pct:.0f}% de presupuesto el
    personal recibiría en promedio <strong>{alarmas_turno:.1f} alarmas por turno</strong>,
    de las cuales cerca de <strong>{aciertos_turno:.1f} corresponden a sepsis real</strong> —
    el resto exige descartarla clínicamente. Es la forma de traducir un porcentaje abstracto
    a la carga de trabajo real de un turno.</p>""",
    unsafe_allow_html=True,
)
st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Evidencia visual: figuras ya generadas en los notebooks
# ---------------------------------------------------------------------------
st.markdown("### Evidencia adicional")
tab1, tab2, tab3 = st.tabs(["Dos pacientes contrastados", "Dependencia SHAP", "Por qué el techo es bajo"])

with tab1:
    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="card"><div class="card-title">El acierto: paciente detectado a tiempo</div>', unsafe_allow_html=True)
        st.image(ruta_figura("08_shap_waterfall_paciente_detectado"), width="stretch")
        st.markdown("</div>", unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="card"><div class="card-title">El fallo: señal real que no bastó</div>', unsafe_allow_html=True)
        st.image(ruta_figura("08_shap_waterfall_paciente_no_detectado"), width="stretch")
        st.markdown("</div>", unsafe_allow_html=True)
    md(
        """<div class="note">El paciente no detectado tenía <strong>más</strong> señal de
        <code>ICULOS</code> que el acierto y un BUN de 84 mg/dL (fallo renal severo). Tres
        señales de baja vigilancia clínica —FiO₂, lactato y conteo de labs sin medir— pesaron
        más que ese valor anormal.</div>""",
        unsafe_allow_html=True,
    )

with tab2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.image(ruta_figura("08_shap_dependencia_top4_variables"), width="stretch")
    st.markdown("</div>", unsafe_allow_html=True)

with tab3:
    c3, c4 = st.columns(2)
    with c3:
        st.markdown('<div class="card"><div class="card-title">Solapamiento entre sépticos y no sépticos</div>', unsafe_allow_html=True)
        st.image(ruta_figura("08_solapamiento_variables_criticas"), width="stretch")
        st.markdown("</div>", unsafe_allow_html=True)
    with c4:
        st.markdown('<div class="card"><div class="card-title">Todas las variables a la vez (beeswarm)</div>', unsafe_allow_html=True)
        st.image(ruta_figura("08_shap_beeswarm"), width="stretch")
        st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Limitaciones
# ---------------------------------------------------------------------------
st.markdown('<div class="card"><div class="card-title">Limitaciones declaradas</div>', unsafe_allow_html=True)
st.markdown(
    """
- **Precisión baja en el punto de operación clínico**: con 1,69% de prevalencia real, el mejor
  modelo genera cerca de 6,8 falsas alarmas por cada acierto al presupuesto del 5%. Es el costo
  estructural de un problema fuertemente desbalanceado, no una falla de diseño.
- **Dependencia de contexto administrativo**: una parte relevante del desempeño se apoya en
  variables que no viajan a otro hospital (unidad, tiempos de ingreso).
- **El dataset no puede reconstruir la definición clínica completa de sepsis**: faltan la
  escala de Glasgow, la mitad del criterio respiratorio y la dosis de vasopresores del score
  SOFA — un techo de predictibilidad estructural, no solo de modelado.
- **El modelo puede ignorar un valor de laboratorio anormal aislado** si llega acompañado de
  poca intensidad de vigilancia clínica alrededor — aprendió que la frecuencia de medición es
  señal real, y eso a veces juega en contra.
- **Etiqueta truncada tras el evento**: el registro de casi todos los pacientes sépticos se
  corta ~9 horas después del inicio de sepsis.
    """
)
st.markdown("</div>", unsafe_allow_html=True)
