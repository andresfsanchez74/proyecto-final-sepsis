"""Tests de src/serving.py: las dos formas de entrada del simulador del dashboard.

Requieren el modelo entrenado (models/modelos_intra_cohorte.joblib) y la muestra versionada
(data/sample/) — el modelo no viaja en git (ver .gitignore), así que estos tests se saltan
solos si alguien corre `pytest tests/` en un clon que todavía no lo generó (notebook 06).
"""

import pandas as pd
import pytest

import io_datos
import serving
from config import cargar_config

CFG = cargar_config()
MODELO_DISPONIBLE = (CFG["rutas"]["models"] / "modelos_intra_cohorte.joblib").exists()

pytestmark = pytest.mark.skipif(
    not MODELO_DISPONIBLE,
    reason="models/modelos_intra_cohorte.joblib no está generado localmente (ver notebook 06)",
)


def test_predecir_hora_actual_devuelve_proba_valida():
    resultado = serving.predecir_hora_actual({
        "HR": 118, "O2Sat": 91, "Temp": 38.6, "SBP": 88, "MAP": 62, "DBP": 50, "Resp": 26,
        "WBC": 14.2, "Lactate": 3.1, "Age": 67, "Gender": 1, "HospAdmTime": -5,
    }, CFG)

    assert len(resultado) == 1
    proba = resultado["proba_sepsis"].iloc[0]
    assert 0.0 <= proba <= 1.0
    assert resultado["alerta"].iloc[0] == (proba >= serving.cargar_umbral(CFG))


def test_predecir_secuencia_usa_historial_del_paciente():
    """Con varias horas reales de un paciente de data/sample, se predice cada hora."""
    cfg_muestra = cargar_config()
    cfg_muestra["rutas"]["raw"] = cfg_muestra["raiz"] / "data" / "sample" / "sepsis_dataset"
    cohorte = io_datos.cargar_cohorte(cfg_muestra, mostrar_progreso=False)

    manifest = pd.read_csv(cfg_muestra["raiz"] / "data" / "sample" / "manifest.csv")
    pid_septico = manifest.loc[manifest["septico"] == 1, "pid"].iloc[0]
    paciente = cohorte[cohorte["pid"] == pid_septico]

    resultado = serving.predecir_secuencia(paciente, CFG)

    assert len(resultado) == len(paciente)
    assert resultado["proba_sepsis"].between(0.0, 1.0).all()
    assert list(resultado["ICULOS"]) == sorted(resultado["ICULOS"])


def test_modelo_inexistente_lanza_error_claro():
    with pytest.raises(ValueError, match="no existe"):
        serving.cargar_paquete(CFG, nombre="ModeloQueNoExiste")
