"""Tests del utility score oficial (src/utility.py).

No dependen del dataset: construyen cohortes sintéticas pequeñas para verificar las dos
propiedades que definen la métrica (ver docstring de utility_normalizado) y un cálculo
manual simple de utilidad_paciente.
"""

import pandas as pd

from utility import utilidad_paciente, utility_normalizado

PARAMS = dict(dt_early=-12, dt_optimal=-6, dt_late=3, max_u_tp=1, min_u_fn=-2, u_fp=-0.05, u_tn=0)


def _cohorte_sintetica():
    """Un paciente no séptico (20h) y uno séptico (30h, positivo desde la hora 20)."""
    filas = []
    for h in range(20):
        filas.append({"pid": "no_septico", "hora": h, "SepsisLabel": 0})
    for h in range(30):
        filas.append({"pid": "septico", "hora": h, "SepsisLabel": 1 if h >= 20 else 0})
    return pd.DataFrame(filas)


def test_utilidad_paciente_falsas_alarmas():
    """Paciente no séptico, alarma en todas las horas: cada hora cuesta u_fp."""
    labels = [0, 0, 0]
    predictions = [1, 1, 1]
    assert utilidad_paciente(labels, predictions, **PARAMS) == 3 * PARAMS["u_fp"]


def test_utilidad_paciente_sin_alarmas_no_septico():
    """Paciente no séptico, nunca se alarma: utilidad neutra (u_tn = 0)."""
    labels = [0, 0, 0, 0]
    predictions = [0, 0, 0, 0]
    assert utilidad_paciente(labels, predictions, **PARAMS) == 0.0


def test_utility_normalizado_predictor_perfecto():
    """Alertar exactamente donde alerta la etiqueta real siempre normaliza a 1.0."""
    df = _cohorte_sintetica()
    df["pred_perfecta"] = df["SepsisLabel"]
    score = utility_normalizado(df, "pred_perfecta", PARAMS, col_paciente="pid")
    assert score == 1.0


def test_utility_normalizado_inaccion():
    """No alertar nunca a nadie es, por definición, el 0.0 de la escala normalizada."""
    df = _cohorte_sintetica()
    df["nunca_alerta"] = 0
    score = utility_normalizado(df, "nunca_alerta", PARAMS, col_paciente="pid")
    assert score == 0.0
