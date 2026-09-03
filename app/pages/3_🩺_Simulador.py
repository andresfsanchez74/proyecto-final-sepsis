"""Página 4 · Simulador de predicción.

La única página que ejecuta el modelo real sobre datos nuevos — el argumento de fondo
para haber elegido Streamlit en vez de Power BI. Dos modos: un instante sin historial
(formulario) o una secuencia completa (paciente de muestra o CSV subido).
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
    cargar_modelo, encabezado, estilizar, inject_css, leer_paciente_muestra, md, obtener_cfg,
)

st.set_page_config(page_title="Sepsis UCI · Simulador", page_icon="🩺", layout="wide")
inject_css()

encabezado(
    "Página 4 de 5",
    "Simulador de predicción",
    "Corre el modelo campeón (CatBoost) sobre un paciente nuevo, en vivo. Esto es lo que "
    "un formulario clínico o un archivo de monitoreo real alimentarían en producción.",
)

cfg = obtener_cfg()
paquete, umbral, error_modelo = cargar_modelo()

if paquete is None:
    aviso_modelo_faltante(error_modelo)
    st.stop()


@st.cache_resource(show_spinner=False)
def _explainer(_paquete):
    import shap
    return shap.TreeExplainer(_paquete["modelo"])


def _matriz_de(fila: dict) -> pd.DataFrame:
    fila_completa = {
        "pid": "simulado", "hosp": fila.get("hosp", "A"), "ICULOS": fila.get("ICULOS", 1),
        "SepsisLabel": 0,
        **{c: fila.get(c) for c in features.CLINICAS},
        **{c: fila.get(c) for c in cfg["datos"]["cols_demograficas"]},
    }
    return features.construir_matriz(pd.DataFrame([fila_completa]), cfg)


def _grafico_semaforo(proba: float, umbral: float):
    color = COLOR_SEPSIS if proba >= umbral else PALETTE["green"]
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=proba * 100,
        number={"suffix": "%", "font": {"size": 40, "family": "Archivo, sans-serif"}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": PALETTE["ink_soft"]},
            "bar": {"color": color, "thickness": 0.35},
            "bgcolor": PALETTE["beige"],
            "steps": [{"range": [0, umbral * 100], "color": PALETTE["beige"]},
                      {"range": [umbral * 100, 100], "color": PALETTE["orange_wash"]}],
            "threshold": {"line": {"color": PALETTE["ink"], "width": 3}, "thickness": 0.9, "value": umbral * 100},
        },
    ))
    return estilizar(fig, height=260, legend=False)


def _waterfall_shap(explainer, X_row: pd.DataFrame, base_value: float, n=10):
    sv = explainer.shap_values(X_row)
    sv = np.asarray(sv).reshape(-1)
    s = pd.Series(sv, index=X_row.columns).sort_values(key=np.abs, ascending=False).head(n)
    s = s.iloc[::-1]
    colores = [COLOR_SEPSIS if v > 0 else COLOR_CONTROL for v in s.values]
    fig = go.Figure(go.Bar(
        x=s.values, y=s.index, orientation="h", marker_color=colores,
        text=[f"{v:+.3f}" for v in s.values], textposition="outside",
    ))
    fig.add_vline(x=0, line_color=PALETTE["ink"], line_width=1)
    fig.update_xaxes(title="Aporte al riesgo (SHAP)")
    fig.update_yaxes(title="")
    return estilizar(fig, height=380, legend=False)


modo = st.radio("Modo", ["Instante (formulario)", "Historia (paciente completo)"], horizontal=True)
st.markdown("<br>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# MODO INSTANTE
# ---------------------------------------------------------------------------
if modo == "Instante (formulario)":
    st.markdown('<div class="card"><div class="card-title">Signos vitales y contexto del paciente</div>', unsafe_allow_html=True)
    with st.form("form_instante"):
        f1, f2, f3, f4 = st.columns(4)
        hr = f1.number_input("Frecuencia cardiaca (HR)", 20, 250, 88)
        temp = f2.number_input("Temperatura (°C)", 30.0, 42.0, 37.0, step=0.1)
        resp = f3.number_input("Frecuencia respiratoria (Resp)", 4, 60, 18)
        o2sat = f4.number_input("Saturación O₂ (%)", 50, 100, 97)

        f5, f6, f7, f8 = st.columns(4)
        sbp = f5.number_input("Presión sistólica (SBP)", 40, 250, 118)
        map_ = f6.number_input("Presión arterial media (MAP)", 30, 180, 80)
        dbp = f7.number_input("Presión diastólica (DBP)", 20, 150, 65)
        iculos = f8.number_input("Horas en UCI (ICULOS)", 1, 400, 24)

        f9, f10, f11, f12 = st.columns(4)
        wbc = f9.number_input("Leucocitos (WBC)", 0.0, 50.0, 9.0, step=0.1)
        lactate = f10.number_input("Lactato", 0.0, 20.0, 1.8, step=0.1)
        creat = f11.number_input("Creatinina", 0.0, 15.0, 1.0, step=0.1)
        edad = f12.number_input("Edad", 0, 110, 65)

        f13, f14, f15 = st.columns(3)
        genero = f13.radio("Sexo", ["Mujer", "Hombre"], horizontal=True)
        hosp = f14.radio("Hospital", ["A", "B"], horizontal=True)
        hospadm = f15.number_input("HospAdmTime (h)", -500.0, 100.0, -4.0, step=1.0)

        enviado = st.form_submit_button("Calcular riesgo", width="stretch")

    st.markdown("</div>", unsafe_allow_html=True)

    if enviado:
        valores = {
            "HR": hr, "Temp": temp, "Resp": resp, "O2Sat": o2sat, "SBP": sbp, "MAP": map_, "DBP": dbp,
            "WBC": wbc, "Lactate": lactate, "Creatinine": creat, "Age": edad,
            "Gender": 1 if genero == "Hombre" else 0, "hosp": hosp, "ICULOS": iculos, "HospAdmTime": hospadm,
        }
        matriz = _matriz_de(valores)
        Xrow = matriz[paquete["cols"]]
        from evaluacion import obtener_proba
        proba = float(obtener_proba(paquete, matriz)[0])
        alerta = proba >= umbral

        st.markdown("<br>", unsafe_allow_html=True)
        colg, colw = st.columns([1, 1.6])
        with colg:
            st.markdown('<div class="card"><div class="card-title">Riesgo estimado</div>', unsafe_allow_html=True)
            st.plotly_chart(_grafico_semaforo(proba, umbral), width="stretch")
            pill = f'<span class="pill pill-on">ALERTA</span>' if alerta else '<span class="pill pill-off">sin alerta</span>'
            st.markdown(f"<div style='text-align:center'>{pill}</div>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
        with colw:
            st.markdown('<div class="card"><div class="card-title">Qué empujó esta predicción (SHAP)</div>', unsafe_allow_html=True)
            explainer = _explainer(paquete)
            st.plotly_chart(_waterfall_shap(explainer, Xrow, explainer.expected_value), width="stretch")
            st.markdown("</div>", unsafe_allow_html=True)

        md(
            """<div class="note"><strong>Recordatorio:</strong> esta herramienta es apoyo de
            tamizaje, no un diagnóstico. Los campos que no se llenan se tratan como "no medido"
            (igual que un examen que aún no se pidió), no como cero.</div>""",
            unsafe_allow_html=True,
        )

# ---------------------------------------------------------------------------
# MODO HISTORIA
# ---------------------------------------------------------------------------
else:
    manifest = cargar_manifest_muestra()
    st.markdown('<div class="card"><div class="card-title">Elegir la fuente de datos</div>', unsafe_allow_html=True)
    fuente = st.radio("Fuente", ["Paciente de muestra", "Subir archivo (.psv o .csv)", "Caso de demostración"], horizontal=True)

    df_pac = None
    if fuente == "Caso de demostración":
        demo = manifest[manifest["septico"] == 1].sort_values("pid").iloc[0]
        df_pac = leer_paciente_muestra(demo["pid"], demo["hosp"])
        st.caption(f"Cargado: {demo['pid']} (hospital {demo['hosp']}, séptico) — caso fijo para no improvisar en vivo.")
    elif fuente == "Paciente de muestra":
        etiquetas = manifest.apply(lambda r: f"{r['pid']} · Hospital {r['hosp']} · {'séptico' if r['septico'] else 'sin sepsis'}", axis=1)
        elegido = st.selectbox("Paciente", list(etiquetas))
        pid_sel = manifest["pid"].iloc[list(etiquetas).index(elegido)]
        hosp_sel = manifest["hosp"].iloc[list(etiquetas).index(elegido)]
        df_pac = leer_paciente_muestra(pid_sel, hosp_sel)
    else:
        archivo = st.file_uploader("Archivo con columnas como un .psv consolidado (separador | o ,)", type=["psv", "csv"])
        if archivo is not None:
            sep = "|" if archivo.name.endswith(".psv") else ","
            df_pac = pd.read_csv(archivo, sep=sep)
            if "pid" not in df_pac.columns:
                df_pac.insert(0, "pid", "subido")
            if "hosp" not in df_pac.columns:
                df_pac.insert(1, "hosp", "A")
    st.markdown("</div>", unsafe_allow_html=True)

    if df_pac is not None:
        import serving
        pred = serving.predecir_secuencia(df_pac, cfg, umbral=umbral)
        onset = df_pac.loc[df_pac["SepsisLabel"] == 1, "ICULOS"].min() if "SepsisLabel" in df_pac else np.nan

        ultima = pred.iloc[-1]
        c1, c2, c3 = st.columns(3)
        c1.metric("Riesgo en la última hora", f"{ultima['proba_sepsis']*100:.1f} %")
        c2.metric("Horas con alerta activa", f"{int(pred['alerta'].sum())} / {len(pred)}")
        c3.metric("Inicio real de sepsis", f"hora {onset:.0f}" if pd.notna(onset) else "sin dato / no desarrolla")

        st.markdown('<div class="card"><div class="card-title">Curva de riesgo completa</div>', unsafe_allow_html=True)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=pred["ICULOS"], y=pred["proba_sepsis"], mode="lines",
                                  line=dict(color=COLOR_SEPSIS, width=2.5), fill="tozeroy",
                                  fillcolor="rgba(220,107,24,0.12)"))
        fig.add_hline(y=umbral, line_dash="dot", line_color=PALETTE["ink_soft"],
                      annotation_text="Umbral de alarma", annotation_position="bottom right")
        if pd.notna(onset):
            fig.add_vline(x=onset, line_dash="dash", line_color=PALETTE["ink"], annotation_text="Inicio real")
        fig.update_xaxes(title="Horas desde el ingreso a UCI")
        fig.update_yaxes(title="Probabilidad de sepsis", range=[0, 1])
        st.plotly_chart(estilizar(fig, height=340, legend=False), width="stretch")
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="card"><div class="card-title">Qué empujó el riesgo en la última hora (SHAP)</div>', unsafe_allow_html=True)
        matriz_completa = features.construir_matriz(df_pac, cfg)
        Xrow = matriz_completa[paquete["cols"]].iloc[[-1]]
        explainer = _explainer(paquete)
        st.plotly_chart(_waterfall_shap(explainer, Xrow, explainer.expected_value), width="stretch")
        st.markdown("</div>", unsafe_allow_html=True)
