"""Página 1 · Panorama de la cohorte.

Responde "¿de qué población estamos hablando?". Todo sale de test_scored.parquet
(versionado en git, 8.068 pacientes) para que el dashboard no dependa de archivos
que .gitignore excluye.
"""

import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from theme import (
    COLOR_CONTROL, COLOR_SEPSIS, PALETTE, cargar_pacientes_test, encabezado,
    estilizar, filtros_sidebar, inject_css, md,
)

st.set_page_config(page_title="Sepsis UCI · Panorama", page_icon="🩺", layout="wide")
inject_css()

encabezado(
    "Predicción temprana de sepsis en UCI · Panorama",
    "¿A quién estamos tratando de predecir?",
    "Cohorte del PhysioNet / CinC Challenge 2019, dos hospitales con prácticas clínicas "
    "distintas. Los filtros de la izquierda se aplican a las cuatro páginas del dashboard.",
)

pac_todos = cargar_pacientes_test()
pac = filtros_sidebar(pac_todos)

if pac.empty:
    st.warning("Ningún paciente cumple los filtros seleccionados.")
    st.stop()

# ---------------------------------------------------------------------------
# KPIs
# ---------------------------------------------------------------------------
c1, c2, c3, c4 = st.columns(4)
c1.metric("Pacientes (muestra de test)", f"{len(pac):,}")
c2.metric("Desarrollan sepsis", f"{pac['sepsis'].mean()*100:.2f} %")
c3.metric("Estancia mediana", f"{pac['los'].median():.0f} h")
mediana_onset = pac.loc[pac["sepsis"] == 1, "hora_inicio_sepsis"].median()
c4.metric("Hora mediana de inicio", f"{mediana_onset:.0f} h" if pac["sepsis"].sum() else "—")

st.markdown("<br>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Fila 1: prevalencia por hospital · demografía
# ---------------------------------------------------------------------------
col1, col2 = st.columns([1, 1.4])

with col1:
    st.markdown('<div class="card"><div class="card-title">Prevalencia de sepsis por hospital</div>', unsafe_allow_html=True)
    prev = pac.groupby("hosp")["sepsis"].mean().mul(100).reindex(["A", "B"])
    fig = go.Figure(go.Bar(
        x=prev.index, y=prev.values, marker_color=[COLOR_SEPSIS, COLOR_CONTROL],
        text=[f"{v:.2f}%" for v in prev.values], textposition="outside",
        width=0.5,
    ))
    fig.update_yaxes(title="% de pacientes con sepsis", rangemode="tozero")
    fig.update_xaxes(title="Hospital")
    st.plotly_chart(estilizar(fig, height=320, legend=False), width="stretch")
    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    st.markdown('<div class="card"><div class="card-title">Distribución de edad, según desenlace</div>', unsafe_allow_html=True)
    pac_g = pac.assign(desenlace=pac["sepsis"].map({0: "No séptico", 1: "Séptico"}))
    fig = px.histogram(
        pac_g, x="edad", color="desenlace", barmode="overlay", histnorm="probability density",
        color_discrete_map={"No séptico": COLOR_CONTROL, "Séptico": COLOR_SEPSIS}, nbins=35,
        opacity=0.65,
    )
    fig.update_xaxes(title="Edad (años)")
    fig.update_yaxes(title="Densidad")
    fig.update_layout(legend_title_text="")
    st.plotly_chart(estilizar(fig, height=320), width="stretch")
    st.markdown("</div>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Fila 2: duración de estancia · momento del inicio
# ---------------------------------------------------------------------------
col3, col4 = st.columns(2)

with col3:
    st.markdown('<div class="card"><div class="card-title">Duración de estancia: la caja es casi igual, la cola no</div>', unsafe_allow_html=True)
    fig = px.box(
        pac_g, x="desenlace", y="los", color="desenlace",
        color_discrete_map={"No séptico": COLOR_CONTROL, "Séptico": COLOR_SEPSIS},
        category_orders={"desenlace": ["No séptico", "Séptico"]},
    )
    fig.update_yaxes(title="Horas de estancia en UCI")
    fig.update_xaxes(title="")
    fig.update_layout(showlegend=False)
    st.plotly_chart(estilizar(fig, height=340, legend=False), width="stretch")
    st.markdown("</div>", unsafe_allow_html=True)

with col4:
    st.markdown('<div class="card"><div class="card-title">¿En qué momento de la estancia se declara la sepsis?</div>', unsafe_allow_html=True)
    septicos = pac[pac["sepsis"] == 1]
    if len(septicos):
        fig = px.histogram(
            septicos, x="hora_inicio_sepsis", color="hosp", nbins=40,
            color_discrete_map={"A": COLOR_SEPSIS, "B": COLOR_CONTROL},
        )
        fig.update_xaxes(title="Hora de estancia (ICULOS)", range=[0, 250])
        fig.update_yaxes(title="Nº de pacientes")
        fig.update_layout(legend_title_text="Hospital")
        st.plotly_chart(estilizar(fig, height=340), width="stretch")
    else:
        st.info("No hay pacientes sépticos en el filtro actual.")
    st.markdown("</div>", unsafe_allow_html=True)

md(
    """
    <div class="note">
    <strong>Lectura rápida:</strong> la edad casi no separa a los grupos y la mediana de estancia
    tampoco — la sepsis no alarga la estancia típica, alarga su cola. La señal real está en la
    fisiología hora a hora, no en quién es el paciente. Ver <em>Explorador de pacientes</em> para
    bajar al nivel individual.
    </div>
    """,
    unsafe_allow_html=True,
)
