"""Identidad visual y carga de datos compartida por las 5 páginas del dashboard.

Centraliza tres cosas para que ninguna página se desincronice de las demás:
- la paleta y el estilo de los gráficos Plotly (blanco / beige / naranja),
- los loaders cacheados (los mismos archivos no se releen en cada rerun de Streamlit),
- el acceso al modelo campeón, con degradación explícita si no está disponible en la
  máquina (models/*.joblib está en .gitignore — no todas las máquinas lo tendrán).
"""

import sys
import textwrap
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# app/ vive junto a src/ en la raíz del repo
sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))

from config import cargar_config  # noqa: E402

PALETTE = {
    "paper": "#FFFFFF",
    "ground": "#FAF5EC",
    "beige": "#F3E9D8",
    "beige_deep": "#E8DAC2",
    "ink": "#2A241D",
    "ink_soft": "#564B3F",
    "muted": "#8B7D6B",
    "rule": "#E2D5BE",
    "orange": "#DC6B18",
    "orange_deep": "#A94D0C",
    "orange_wash": "#FBEAD6",
    "slate": "#5B7A8C",
    "slate_wash": "#E8EEF1",
    "green": "#4A7C32",
    "red": "#B23A2B",
}

# Dos colores semánticos reutilizados en todo el dashboard: naranja para "positivo /
# hospital A / alerta", pizarra para "negativo / hospital B / control". Mantiene la
# identidad de dos tonos en vez de una paleta categórica genérica.
COLOR_SEPSIS = PALETTE["orange"]
COLOR_CONTROL = PALETTE["slate"]
COLOR_A = PALETTE["orange"]
COLOR_B = PALETTE["slate"]

MODELOS_TABULARES = ["CatBoost", "XGBoost", "HistGradientBoosting", "LightGBM", "LogReg"]
MODELO_CAMPEON = "CatBoost"


def inject_css():
    # Dos llamadas separadas a propósito: si <style> no es la primera etiqueta del bloque,
    # Markdown lo trata como bloque HTML genérico (termina en la primera línea en blanco) en
    # vez de bloque tipo <style> (termina solo en </style>) -- una línea en blanco en medio del
    # CSS bastaba para que el resto del bloque se mostrara como texto plano.
    st.markdown(
        '<link rel="preconnect" href="https://fonts.googleapis.com">'
        '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>'
        '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Archivo:wght@500;600;700&family=IBM+Plex+Mono:wght@400;500;600&family=Karla:wght@400;500;600;700&display=swap">',
        unsafe_allow_html=True,
    )
    st.markdown(
        textwrap.dedent(
            """\
        <style>
        html, body, [class*="css"] { font-family: 'Karla', sans-serif; }
        h1, h2, h3, .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
            font-family: 'Archivo', sans-serif !important;
            letter-spacing: -0.01em;
            color: #2A241D;
        }
        [data-testid="stAppViewContainer"] { background: #FAF5EC; }
        [data-testid="stSidebar"] { background: #F3E9D8; border-right: 1px solid #E2D5BE; }
        [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
            font-family: 'Archivo', sans-serif !important;
        }
        [data-testid="stHeader"] { background: rgba(0,0,0,0); }

        .eyebrow {
            font-family: 'IBM Plex Mono', monospace;
            font-size: 11.5px; letter-spacing: 0.14em; text-transform: uppercase;
            color: #A94D0C; margin-bottom: 4px;
        }

        [data-testid="stMetric"] {
            background: #FFFFFF;
            border: 1px solid #E2D5BE;
            border-left: 3px solid #DC6B18;
            border-radius: 4px;
            padding: 14px 18px 12px;
        }
        [data-testid="stMetricLabel"] {
            font-family: 'IBM Plex Mono', monospace !important;
            font-size: 10.5px !important; letter-spacing: 0.08em; text-transform: uppercase;
            color: #8B7D6B !important;
        }
        [data-testid="stMetricValue"] {
            font-family: 'Archivo', sans-serif !important;
            color: #2A241D !important;
        }

        .card {
            background: #FFFFFF; border: 1px solid #E2D5BE; border-radius: 4px;
            padding: 18px 22px; margin-bottom: 14px;
        }
        .card-title {
            font-family: 'Archivo', sans-serif; font-weight: 600; font-size: 15px;
            margin-bottom: 6px; color: #2A241D;
        }
        .note {
            background: #FBEAD6; border-left: 3px solid #A94D0C;
            padding: 12px 16px; border-radius: 2px; font-size: 14.5px; color: #564B3F;
        }
        .pill {
            display: inline-block; font-family: 'IBM Plex Mono', monospace; font-size: 11px;
            padding: 3px 9px; border-radius: 20px; margin-right: 6px;
        }
        .pill-on { background: #DC6B18; color: #FFFFFF; }
        .pill-off { background: #E8DAC2; color: #564B3F; }

        [data-testid="stDataFrame"] { border: 1px solid #E2D5BE; border-radius: 4px; }
        hr { border-color: #E2D5BE; }
        </style>
        """
        ),
        unsafe_allow_html=True,
    )


def md(texto: str, **kwargs):
    """st.markdown que corrige la indentación de Python antes de renderizar.

    Una cadena triple-comillada dentro de una función queda con la sangría del código
    fuente; Markdown interpreta 4+ espacios de indentación como bloque de código, así
    que sin este ajuste el HTML se muestra como texto plano en vez de renderizarse.
    """
    st.markdown(textwrap.dedent(texto), **kwargs)


def estilizar(fig: go.Figure, height: int = 420, legend: bool = True, titulo: str | None = None) -> go.Figure:
    """Aplica el estilo visual único del dashboard a cualquier figura Plotly."""
    if titulo:
        fig.update_layout(title=dict(text=titulo, font=dict(family="Archivo, sans-serif", size=16, color=PALETTE["ink"])))
    fig.update_layout(
        height=height,
        paper_bgcolor=PALETTE["paper"],
        plot_bgcolor=PALETTE["paper"],
        font=dict(family="Karla, sans-serif", color=PALETTE["ink"], size=13),
        margin=dict(l=8, r=8, t=48 if titulo else 16, b=8),
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="left", x=0,
                    bgcolor="rgba(0,0,0,0)") if legend else dict(),
        showlegend=legend,
        hoverlabel=dict(bgcolor=PALETTE["ink"], font_color=PALETTE["paper"], font_family="IBM Plex Mono, monospace"),
        colorway=[PALETTE["orange"], PALETTE["slate"], PALETTE["orange_deep"], PALETTE["muted"]],
    )
    fig.update_xaxes(gridcolor=PALETTE["rule"], zerolinecolor=PALETTE["rule"], linecolor=PALETTE["rule"])
    fig.update_yaxes(gridcolor=PALETTE["rule"], zerolinecolor=PALETTE["rule"], linecolor=PALETTE["rule"])
    return fig


# ---------------------------------------------------------------------------
# Config y datos
# ---------------------------------------------------------------------------

@st.cache_resource
def obtener_cfg():
    return cargar_config()


@st.cache_data
def cargar_test_scored() -> pd.DataFrame:
    cfg = obtener_cfg()
    return pd.read_parquet(cfg["rutas"]["dashboard"] / "test_scored.parquet")


@st.cache_data
def cargar_pacientes_test() -> pd.DataFrame:
    """Agrega test_scored a nivel paciente: una fila por pid con sus atributos fijos."""
    df = cargar_test_scored()
    onset = df.loc[df["SepsisLabel"] == 1].groupby("pid")["ICULOS"].min()
    pac = df.groupby("pid").agg(
        hosp=("hosp", "first"), edad=("Age", "first"), genero=("Gender", "first"),
        sepsis=("SepsisLabel", "max"), los=("ICULOS", "max"),
    )
    pac["hora_inicio_sepsis"] = onset.reindex(pac.index)
    return pac


@st.cache_data
def cargar_importancia_shap() -> pd.DataFrame:
    cfg = obtener_cfg()
    return pd.read_csv(cfg["rutas"]["dashboard"] / "importancia_shap.csv")


@st.cache_data
def cargar_manifest_muestra() -> pd.DataFrame:
    cfg = obtener_cfg()
    return pd.read_csv(cfg["raiz"] / "data" / "sample" / "manifest.csv", dtype={"pid": str})


@st.cache_data
def leer_paciente_muestra(pid: str, hosp: str) -> pd.DataFrame:
    cfg = obtener_cfg()
    subcarpeta = cfg["datos"]["hospitales"][hosp]
    ruta = cfg["raiz"] / "data" / "sample" / "sepsis_dataset" / subcarpeta / f"{pid}.psv"
    d = pd.read_csv(ruta, sep="|")
    d.insert(0, "pid", pid)
    d.insert(1, "hosp", hosp)
    return d


@st.cache_data
def ruta_figura(nombre: str) -> str:
    cfg = obtener_cfg()
    return str(cfg["rutas"]["figures"] / f"{nombre}.png")


@st.cache_resource(show_spinner="Cargando modelo campeón…")
def cargar_modelo(nombre: str = MODELO_CAMPEON):
    """Carga el modelo entrenado y su umbral. Devuelve (paquete, umbral, error)."""
    import serving
    cfg = obtener_cfg()
    try:
        paquete = serving.cargar_paquete(cfg, nombre)
        umbral = serving.cargar_umbral(cfg, nombre)
        return paquete, umbral, None
    except FileNotFoundError as e:
        return None, None, str(e)


def aviso_modelo_faltante(detalle: str):
    md(
        f"""
        <div class="note">
        <strong>El modelo entrenado no está disponible en esta máquina.</strong>
        <code>models/*.joblib</code> no se versiona en git (pesa ~16 MB y se regenera con el
        notebook 06). Para habilitar esta página, ejecuta
        <code>notebooks/06_modelado_baselines.ipynb</code> localmente, o copia el archivo
        desde la máquina donde se entrenó.<br><br>
        <span style="font-family:'IBM Plex Mono',monospace;font-size:12.5px;opacity:0.75">{detalle}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def encabezado(kicker: str, titulo: str, lede: str):
    st.markdown(f'<p class="eyebrow">{kicker}</p>', unsafe_allow_html=True)
    st.title(titulo)
    st.markdown(f"<p style='font-size:17px;color:#564B3F;max-width:70ch'>{lede}</p>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)


def filtros_sidebar(pac: pd.DataFrame) -> pd.DataFrame:
    """Filtros globales reutilizados en cada página (misma key -> mismo estado)."""
    st.sidebar.markdown("### Filtros")
    hosp_sel = st.sidebar.multiselect("Hospital", ["A", "B"], default=["A", "B"], key="flt_hosp")
    edad_min, edad_max = int(pac["edad"].min()), int(pac["edad"].max())
    rango_edad = st.sidebar.slider("Rango de edad", edad_min, edad_max, (edad_min, edad_max), key="flt_edad")
    genero_sel = st.sidebar.radio("Sexo", ["Todos", "Mujeres", "Hombres"], key="flt_genero", horizontal=True)

    out = pac[pac["hosp"].isin(hosp_sel) & pac["edad"].between(*rango_edad)]
    if genero_sel == "Mujeres":
        out = out[out["genero"] == 0]
    elif genero_sel == "Hombres":
        out = out[out["genero"] == 1]
    return out
