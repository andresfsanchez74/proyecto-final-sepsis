"""Utility score oficial del PhysioNet/CinC Challenge 2019.

Reimplementación de la lógica de `evaluate_sepsis_score.py` del organizador. Es la métrica
que define el problema, y no existe en scikit-learn, así que hay que traerla al proyecto.

La idea clínica: acertar no vale lo mismo en todo momento. Una alerta 6 h antes del inicio
de sepsis vale el máximo (max_u_tp = 1); antes de 12 h no vale nada porque el médico no
tiene por qué creerla todavía; después del inicio la recompensa decae y no detectar castiga
hasta -2. Una falsa alarma cuesta poco (-0.05) pero se paga en cada hora que se mantenga:
así el score captura la fatiga de alarma sin ahogar la sensibilidad.

Esta asimetría es la razón de fondo por la que optimizar accuracy o F1 aquí no tiene sentido.
"""

import numpy as np


def utilidad_paciente(
    labels,
    predictions,
    dt_early=-12,
    dt_optimal=-6,
    dt_late=3.0,
    max_u_tp=1,
    min_u_fn=-2,
    u_fp=-0.05,
    u_tn=0,
) -> float:
    """Utilidad sin normalizar de una secuencia de predicciones para UN paciente."""
    labels = np.asarray(labels)
    predictions = np.asarray(predictions)

    if np.any(labels):
        es_septico = True
        # SepsisLabel ya viene adelantada 6 h por el organizador: la primera etiqueta 1
        # ocurre en t_sepsis + dt_optimal, así que el inicio real se recupera restando.
        t_sepsis = np.argmax(labels) - dt_optimal
    else:
        es_septico = False
        t_sepsis = float("inf")

    # Rectas que interpolan la recompensa/castigo entre los instantes clave
    m_1 = float(max_u_tp) / float(dt_optimal - dt_early)
    b_1 = -m_1 * dt_early
    m_2 = float(-max_u_tp) / float(dt_late - dt_optimal)
    b_2 = -m_2 * dt_late
    m_3 = float(min_u_fn) / float(dt_late - dt_optimal)
    b_3 = -m_3 * dt_optimal

    u = np.zeros(len(labels))
    for t in range(len(labels)):
        if t > t_sepsis + dt_late:
            continue  # pasado el margen tardío ya no se premia ni se castiga
        if es_septico and predictions[t]:
            if t <= t_sepsis + dt_optimal:
                u[t] = max(m_1 * (t - t_sepsis) + b_1, u_fp)
            else:
                u[t] = m_2 * (t - t_sepsis) + b_2
        elif not es_septico and predictions[t]:
            u[t] = u_fp
        elif es_septico and not predictions[t]:
            if t > t_sepsis + dt_optimal:
                u[t] = m_3 * (t - t_sepsis) + b_3
        else:
            u[t] = u_tn

    return float(np.sum(u))


def utility_normalizado(df, col_pred, cfg_utility, col_paciente="pid", col_target="SepsisLabel") -> float:
    """Score oficial normalizado sobre una cohorte completa.

    Se normaliza contra dos referencias para que el número sea interpretable:
      1.0 = predictor perfecto (alerta exactamente en la ventana óptima de cada séptico)
      0.0 = no alertar nunca a nadie
    Valores negativos significan que el modelo es peor que quedarse callado.
    """
    p = {k: cfg_utility[k] for k in
         ["dt_early", "dt_optimal", "dt_late", "max_u_tp", "min_u_fn", "u_fp", "u_tn"]}

    obtenida = inaccion = perfecta = 0.0
    for _, g in df.groupby(col_paciente, sort=False):
        y = g[col_target].to_numpy()
        obtenida += utilidad_paciente(y, g[col_pred].to_numpy(), **p)
        inaccion += utilidad_paciente(y, np.zeros(len(y)), **p)
        perfecta += utilidad_paciente(y, y, **p)

    return (obtenida - inaccion) / (perfecta - inaccion)
