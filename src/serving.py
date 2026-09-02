"""Predicción sobre datos nuevos, para el simulador del dashboard.

Reutiliza exactamente el mismo pipeline de features que el entrenamiento
(`features.construir_matriz`) y la misma lógica de preprocesamiento por modelo
(`evaluacion.obtener_proba`) — nunca una versión simplificada aparte, para que el simulador
prediga con las mismas reglas que se evaluaron en los notebooks 06-08, no con un atajo que
podría desincronizarse de ellas.

Dos formas de entrada, según cuánta historia tiene el usuario:

- `predecir_secuencia`: una tabla con varias horas de un paciente (ideal — igual que un `.psv`
  real, permite calcular las features de ventana rodante y desviación del basal).
- `predecir_hora_actual`: solo los valores de este instante, sin historial. Se arma un
  "paciente" de una sola fila; `construir_matriz` sigue funcionando, pero las features que
  dependen de historia caen a su valor neutro (antigüedad 0, sin tendencia, delta del basal
  0) porque no hay horas previas con las que compararse — el modelo predice igual, con menos
  información real.
"""

import joblib
import pandas as pd

import features
from evaluacion import obtener_proba

MODELO_CAMPEON = "CatBoost"


def cargar_paquete(cfg, nombre=MODELO_CAMPEON):
    """Carga el modelo entrenado (y su preprocesamiento, si lo necesita) desde el .joblib."""
    paquete_completo = joblib.load(cfg["rutas"]["models"] / "modelos_intra_cohorte.joblib")
    if nombre not in paquete_completo["modelos"]:
        disponibles = list(paquete_completo["modelos"])
        raise ValueError(f"Modelo '{nombre}' no existe. Opciones: {disponibles}")
    return paquete_completo["modelos"][nombre]


def cargar_umbral(cfg, nombre=MODELO_CAMPEON):
    """Umbral de decisión ya elegido en el notebook 07 (presupuesto de alarma del 5%)."""
    panel = pd.read_csv(cfg["rutas"]["models"] / "resultados_utility_umbral.csv")
    fila = panel.loc[panel["modelo"] == nombre]
    if fila.empty:
        raise ValueError(f"No hay umbral guardado para '{nombre}' en resultados_utility_umbral.csv")
    return float(fila["umbral"].iloc[0])


def predecir_secuencia(df_horas: pd.DataFrame, cfg, nombre=MODELO_CAMPEON, umbral=None) -> pd.DataFrame:
    """Predice sobre una o más horas reales/simuladas de uno o más pacientes.

    `df_horas` debe traer las mismas columnas que un `.psv` consolidado: `pid`, `hosp`,
    `ICULOS`, las demográficas (`Age`, `Gender`, `Unit1`, `Unit2`, `HospAdmTime`) y los
    signos vitales/laboratorios disponibles (`features.CLINICAS`); lo que falte se trata
    como no medido, igual que en el dataset real.
    """
    df_horas = df_horas.copy()
    if "SepsisLabel" not in df_horas:
        df_horas["SepsisLabel"] = 0  # no se usa como feature; construir_matriz solo la pasa de largo

    matriz = features.construir_matriz(df_horas, cfg)

    paquete = cargar_paquete(cfg, nombre)
    X = matriz[paquete["cols"]]
    proba = obtener_proba(paquete, X)
    umbral = cargar_umbral(cfg, nombre) if umbral is None else umbral

    salida = matriz[["pid", "hosp", "ICULOS"]].copy()
    salida["proba_sepsis"] = proba
    salida["alerta"] = proba >= umbral
    return salida


def predecir_hora_actual(valores: dict, cfg, nombre=MODELO_CAMPEON, umbral=None) -> pd.DataFrame:
    """Predice a partir de un único instante, sin historial previo (formulario simple).

    `valores` es un dict con lo que el usuario llenó en el formulario: cualquier subconjunto
    de `features.CLINICAS` (signos vitales/laboratorios) y de las demográficas. Lo que no se
    pase queda como no medido (`None` -> NaN), igual que un paciente al que aún no le han
    hecho ese examen.
    """
    fila = {
        "pid": "simulado",
        "hosp": valores.get("hosp", "A"),
        "ICULOS": valores.get("ICULOS", 1),
        **{col: valores.get(col) for col in features.CLINICAS},
        **{col: valores.get(col) for col in cfg["datos"]["cols_demograficas"]},
    }
    df_horas = pd.DataFrame([fila])
    return predecir_secuencia(df_horas, cfg, nombre, umbral)
