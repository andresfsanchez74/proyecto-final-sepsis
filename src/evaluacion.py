"""Particiones sin fuga y métricas apropiadas para un problema desbalanceado y temporal.

Dos ideas gobiernan este módulo:

**La partición es por paciente, nunca por fila.** Un `train_test_split` normal repartiría las
horas de un mismo paciente entre train y test. Como las horas consecutivas de una persona son
casi idénticas, el modelo estaría prediciendo en test a pacientes que ya vio: el AUC subiría
varios puntos sin que el modelo haya aprendido nada generalizable. Todo aquí agrupa por `pid`.

**El accuracy no sirve.** Con 1,85 % de horas positivas, un modelo que responda "no" siempre
acierta el 98,15 % de las veces y es clínicamente inútil. Por eso reportamos AUPRC (sensible al
desbalance), el utility score oficial (sensible al *momento* del acierto) y la sensibilidad a
una tasa de alarma fija, que es la pregunta que de verdad haría un jefe de UCI.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

from utility import utility_normalizado


def obtener_proba(paquete, X):
    """Probabilidad de la clase positiva para un modelo guardado por `06_modelado_baselines`.

    Centraliza el único detalle que varía entre los cinco modelos: la regresión logística
    guarda su `imputer`/`scaler` junto al modelo (es la única que no acepta NaN); los cuatro
    de árboles no. Se usa igual en los notebooks 06, 07 y 08 para no repetir este `if` cuatro
    veces con la oportunidad de que alguna copia se desincronice de las demás.
    """
    Xp = X[paquete["cols"]]
    if "imputer" in paquete:
        Xp = paquete["scaler"].transform(paquete["imputer"].transform(Xp))
    return paquete["modelo"].predict_proba(Xp)[:, 1]


def split_por_paciente(df, test_size=0.2, semilla=42):
    """Divide en train/test manteniendo íntegro a cada paciente en un solo lado."""
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=semilla)
    idx_train, idx_test = next(splitter.split(df, groups=df["pid"]))
    return df.iloc[idx_train].copy(), df.iloc[idx_test].copy()


def split_por_hospital(df, hospital_train="A"):
    """Validación externa: entrenar en un hospital y evaluar en el otro.

    Es la prueba más exigente y la más parecida a desplegar el modelo en un sitio nuevo:
    poblaciones distintas, prácticas de medición distintas y prevalencias distintas.
    """
    return (df[df["hosp"] == hospital_train].copy(),
            df[df["hosp"] != hospital_train].copy())


def folds_agrupados(df, n_splits=5):
    """Folds de validación cruzada que nunca parten a un paciente entre train y validación."""
    return list(GroupKFold(n_splits=n_splits).split(df, groups=df["pid"]))


def sensibilidad_a_tasa_alarma(y_true, y_score, tasa_alarma):
    """Sensibilidad alcanzable si solo se tolera alarmar en un `tasa_alarma` de las horas.

    Traduce el desempeño a la restricción real de una UCI: el personal solo atenderá un número
    limitado de alertas por turno antes de empezar a ignorarlas. Fijamos ese presupuesto y
    preguntamos cuántas sepsis se detectan dentro de él.
    """
    umbral = np.quantile(y_score, 1 - tasa_alarma)
    alarma = y_score >= umbral
    positivos = np.asarray(y_true) == 1
    return float(alarma[positivos].mean()), float(umbral)


def horas_de_anticipacion(df, col_score, umbral, col_paciente="pid", col_target="SepsisLabel"):
    """Cuántas horas antes de la primera etiqueta positiva se dispara la primera alarma.

    La métrica que le importa al clínico: no basta con detectar, hay que detectar **a tiempo**
    para que quede margen de actuar. Solo se calcula en pacientes sépticos efectivamente
    detectados; los no detectados se contabilizan aparte como fallos.
    """
    anticipaciones, no_detectados = [], 0

    for _, g in df[df[col_target].groupby(df[col_paciente], observed=True).transform("max") == 1] \
            .groupby(col_paciente, observed=True, sort=False):
        t_label = g.loc[g[col_target] == 1, "ICULOS"].min()
        alarmas = g.loc[g[col_score] >= umbral, "ICULOS"]
        alarmas_previas = alarmas[alarmas <= t_label]

        if len(alarmas_previas):
            anticipaciones.append(t_label - alarmas_previas.min())
        else:
            no_detectados += 1

    s = pd.Series(anticipaciones, dtype="float64")
    return {
        "detectados": len(anticipaciones),
        "no_detectados_antes_del_evento": no_detectados,
        "anticipacion_mediana_h": float(s.median()) if len(s) else np.nan,
        "anticipacion_p25_h": float(s.quantile(0.25)) if len(s) else np.nan,
        "anticipacion_p75_h": float(s.quantile(0.75)) if len(s) else np.nan,
    }


def evaluar(df, y_score, cfg, nombre="modelo", umbral=None):
    """Panel completo de métricas para un conjunto de predicciones sobre una cohorte."""
    d = df.copy()
    d["score"] = np.asarray(y_score)
    y = d["SepsisLabel"].to_numpy()

    tasa = cfg["evaluacion"]["tasa_alarma_objetivo"]
    sens, umbral_tasa = sensibilidad_a_tasa_alarma(y, d["score"], tasa)
    umbral = umbral_tasa if umbral is None else umbral

    d["pred"] = (d["score"] >= umbral).astype("int8")

    resultado = {
        "modelo": nombre,
        "AUPRC": average_precision_score(y, d["score"]),
        "AUROC": roc_auc_score(y, d["score"]),
        "utility_normalizado": utility_normalizado(d, "pred", cfg["utility"]),
        "umbral": umbral,
        f"sensibilidad_a_{tasa:.0%}_alarmas": sens,
        "prevalencia_horas": float(y.mean()),
    }
    resultado.update(horas_de_anticipacion(d, "score", umbral))
    return resultado


def barrido_de_umbral(df, y_score, cfg, n_umbrales=40):
    """Recorre umbrales y devuelve la curva de utility, para elegirlo con criterio.

    El 0,5 por defecto de scikit-learn no tiene ningún significado clínico: sale de asumir
    costes simétricos entre error y acierto, que es justo lo contrario de este problema.
    El umbral se elige aquí, maximizando la métrica que sí codifica esos costes.
    """
    d = df.copy()
    d["score"] = np.asarray(y_score)

    umbrales = np.quantile(d["score"], np.linspace(0.5, 0.999, n_umbrales))
    filas = []
    for u in np.unique(umbrales):
        d["pred"] = (d["score"] >= u).astype("int8")
        filas.append({
            "umbral": u,
            "utility": utility_normalizado(d, "pred", cfg["utility"]),
            "tasa_alarma": float(d["pred"].mean()),
        })
    return pd.DataFrame(filas)
