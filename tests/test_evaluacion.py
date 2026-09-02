"""Tests de src/evaluacion.py: la partición sin fuga y la métrica de tasa de alarma fija."""

import numpy as np
import pandas as pd

from evaluacion import sensibilidad_a_tasa_alarma, split_por_paciente


def test_split_por_paciente_no_mezcla_pacientes():
    """Ningún pid puede aparecer a la vez en train y en test (la fuga que motiva el módulo)."""
    filas = [{"pid": pid, "hora": h} for pid in range(50) for h in range(10)]
    df = pd.DataFrame(filas)

    train, test = split_por_paciente(df, test_size=0.2, semilla=42)

    pacientes_train = set(train["pid"])
    pacientes_test = set(test["pid"])
    assert pacientes_train.isdisjoint(pacientes_test)
    assert pacientes_train | pacientes_test == set(df["pid"])


def test_sensibilidad_a_tasa_alarma_presupuesto_exacto():
    """Con un presupuesto de alarma del 10% sobre 100 horas, se alarma en las 10 más altas."""
    y_score = np.arange(100)  # scores 0..99, únicos
    y_true = np.zeros(100)
    y_true[95:] = 1  # las 5 horas de mayor score son las positivas reales

    sensibilidad, umbral = sensibilidad_a_tasa_alarma(y_true, y_score, tasa_alarma=0.10)

    assert sensibilidad == 1.0  # los 5 positivos caen dentro del 10% de mayor score
    assert umbral == np.quantile(y_score, 0.90)
