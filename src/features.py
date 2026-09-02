"""Ingeniería de características temporales, estrictamente causal.

Regla que gobierna todo este módulo: **para predecir la hora t solo se puede usar información
disponible hasta la hora t, inclusive**. Todo se calcula dentro de cada paciente (`groupby(pid)`)
y siempre hacia adelante en el tiempo. Nunca `bfill`, nunca `interpolate`, nunca un estadístico
calculado sobre la estancia completa.

Por qué es tan estricto: en la hora 10 de un paciente, un `interpolate()` rellenaría el hueco
usando el laboratorio de la hora 30. Ese valor todavía no existe cuando el modelo debe decidir.
El modelo aprendería del futuro, las métricas saldrían excelentes y en producción no serviría.

El módulo produce cuatro familias de features, cada una con una hipótesis clínica detrás:

1. **Valor arrastrado + antigüedad** — el último valor conocido y hace cuánto se midió.
2. **Indicadores de medición** — que se pida un examen es una decisión del médico, no un dato
   perdido al azar: es información sobre su nivel de sospecha.
3. **Dinámica en ventanas** — la tendencia anticipa mejor que el nivel absoluto.
4. **Desviación del basal propio** — una FC de 100 no significa lo mismo en todos los pacientes.
"""

import numpy as np
import pandas as pd

# Signos vitales: registro casi continuo por monitor. Nulos ~10-30 %.
VITALES = ["HR", "O2Sat", "Temp", "SBP", "MAP", "DBP", "Resp", "EtCO2"]

# Laboratorios: se solicitan puntualmente. Nulos >90 %; aquí la medición misma es la señal.
LABORATORIOS = [
    "BaseExcess", "HCO3", "FiO2", "pH", "PaCO2", "SaO2", "AST", "BUN", "Alkalinephos",
    "Calcium", "Chloride", "Creatinine", "Bilirubin_direct", "Glucose", "Lactate",
    "Magnesium", "Phosphate", "Potassium", "Bilirubin_total", "TroponinI", "Hct",
    "Hgb", "PTT", "WBC", "Fibrinogen", "Platelets",
]

CLINICAS = VITALES + LABORATORIOS


def ordenar(df: pd.DataFrame) -> pd.DataFrame:
    """Garantiza el orden temporal dentro de cada paciente; sin esto nada causal es válido."""
    return df.sort_values(["pid", "ICULOS"], kind="stable").reset_index(drop=True)


def valor_arrastrado(df: pd.DataFrame, cols: list[str], max_horas: int) -> pd.DataFrame:
    """Último valor conocido de cada variable, con tope de antigüedad, y su edad en horas.

    El `ffill` codifica el supuesto clínico "sin noticias, sin cambios": si el potasio de hace
    dos horas era normal, probablemente lo siga siendo. Ese supuesto se degrada con el tiempo,
    y por eso se corta en `max_horas`: un lactato de hace tres días no describe al paciente de
    ahora, y arrastrarlo sería inventar una medición que nadie hizo.

    La columna de antigüedad es la contraparte honesta del arrastre: le dice al modelo cuánto
    puede confiar en cada valor, en vez de presentarle todo como si acabara de medirse.
    """
    g = df.groupby("pid", observed=True, sort=False)
    salida = {}

    for col in cols:
        arrastrado = g[col].ffill()

        # Antigüedad: ICULOS actual menos el ICULOS de la última hora con medición real
        iculos_medido = df["ICULOS"].where(df[col].notna())
        ultima_medicion = iculos_medido.groupby(df["pid"], observed=True, sort=False).ffill()
        edad = df["ICULOS"] - ultima_medicion

        # Se descarta el valor cuya antigüedad supera el tope, pero se conserva la edad:
        # "hace mucho que no se mide" sigue siendo información útil.
        salida[f"{col}_ult"] = arrastrado.where(edad <= max_horas)
        salida[f"{col}_edad"] = edad

    return pd.DataFrame(salida, index=df.index)


def indicadores_medicion(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Marca si la variable se midió en esta hora concreta (missingness informativa).

    En una UCI un laboratorio no falta al azar: falta porque nadie lo pidió. Que en la hora 14
    aparezca un lactato significa que alguien sospechó hipoperfusión justo entonces. Tratar ese
    hueco como "dato perdido" y taparlo con una media borraría la decisión clínica que lo generó.
    """
    return pd.DataFrame(
        {f"{col}_medido": df[col].notna().astype("int8") for col in cols},
        index=df.index,
    )


def carga_de_monitoreo(df: pd.DataFrame, ventanas: list[int]) -> pd.DataFrame:
    """Intensidad de vigilancia: cuánto se está midiendo a este paciente ahora mismo.

    Hipótesis a contrastar en el notebook 02: cuando un paciente empieza a deteriorarse, el
    equipo pide más exámenes y con más frecuencia. Si es cierta, un simple conteo de mediciones
    por hora anticipa el evento sin mirar el valor de ninguna variable clínica.
    """
    n_labs = df[LABORATORIOS].notna().sum(axis=1).astype("float32")
    n_vitales = df[VITALES].notna().sum(axis=1).astype("float32")

    salida = {"n_labs_hora": n_labs, "n_vitales_hora": n_vitales}
    base = pd.DataFrame({"pid": df["pid"], "n_labs": n_labs, "n_vitales": n_vitales}, index=df.index)
    g = base.groupby("pid", observed=True, sort=False)

    # GroupBy.rolling() (no .transform(lambda ...)): calcula la ventana rodante de todos los
    # grupos de una sola pasada vectorizada. Con 40.336 pacientes, `.transform(lambda s: ...)`
    # despacha la llamada en Python grupo por grupo y tarda órdenes de magnitud más.
    for v in ventanas:
        salida[f"n_labs_{v}h"] = g["n_labs"].rolling(v, min_periods=1).sum().reset_index(level=0, drop=True)
        salida[f"n_vitales_{v}h"] = g["n_vitales"].rolling(v, min_periods=1).sum().reset_index(level=0, drop=True)

    return pd.DataFrame(salida, index=df.index)


def dinamica_en_ventana(df_ult: pd.DataFrame, pid: pd.Series, cols: list[str],
                        ventanas: list[int]) -> pd.DataFrame:
    """Estadísticos rodantes y pendiente sobre el valor arrastrado.

    Se calcula sobre la serie ya arrastrada, no sobre la cruda: con >90 % de nulos, una ventana
    de 6 horas sobre la serie cruda estaría casi siempre vacía. La pendiente se aproxima como
    (valor actual − valor de hace v horas) / v, que capta la tendencia sin el coste de ajustar
    una regresión por fila en 1,5 millones de filas.

    `min_periods=1` es deliberado: en las primeras horas de un paciente la ventana está
    incompleta y aun así queremos una estimación, no un nulo. El modelo ya sabe cuántas horas
    lleva el paciente gracias a `ICULOS`.
    """
    # Igual que en carga_de_monitoreo: se usa el accesor GroupBy.rolling() vectorizado, nunca
    # .transform(lambda s: s.rolling(...)) -- esa forma repite un despacho en Python por cada
    # uno de los 40.336 pacientes y por cada una de las ~70 combinaciones variable/ventana/
    # estadístico, lo que en la práctica tardaba más de 100 minutos sobre la cohorte completa.
    salida = {}
    g = df_ult.groupby(pid, observed=True, sort=False)
    for v in ventanas:
        for col in cols:
            c = f"{col}_ult"
            roll = g[c].rolling(v, min_periods=1)
            salida[f"{col}_media_{v}h"] = roll.mean().reset_index(level=0, drop=True)
            salida[f"{col}_std_{v}h"] = roll.std().reset_index(level=0, drop=True)
            salida[f"{col}_min_{v}h"] = roll.min().reset_index(level=0, drop=True)
            salida[f"{col}_max_{v}h"] = roll.max().reset_index(level=0, drop=True)
            salida[f"{col}_pend_{v}h"] = (df_ult[c] - g[c].shift(v)) / v

    return pd.DataFrame(salida, index=df_ult.index)


def desviacion_del_basal(df_ult: pd.DataFrame, pid: pd.Series, cols: list[str],
                         horas_basal: int) -> pd.DataFrame:
    """Cuánto se ha alejado el paciente de su propio punto de partida.

    Una frecuencia cardiaca de 100 es alarmante en alguien que ingresó en 60 y rutinaria en
    alguien que ingresó en 95. Normalizar contra el propio paciente convierte un valor absoluto
    en un cambio relativo, que es lo que realmente vigila el clínico.

    El basal usa solo las primeras `horas_basal` horas de estancia, así que ya está disponible
    en cualquier instante posterior: no viola la causalidad.
    """
    inicio = df_ult.assign(_pid=pid.values, _n=df_ult.groupby(pid, observed=True, sort=False).cumcount())
    primeras = inicio[inicio["_n"] < horas_basal]
    basal = primeras.groupby("_pid", observed=True)[[f"{c}_ult" for c in cols]].mean()

    basal_alineado = basal.reindex(pid.values).to_numpy()
    actual = df_ult[[f"{c}_ult" for c in cols]].to_numpy()

    return pd.DataFrame(
        actual - basal_alineado,
        columns=[f"{c}_delta_basal" for c in cols],
        index=df_ult.index,
    )


def scores_clinicos(df_ult: pd.DataFrame) -> pd.DataFrame:
    """Índices que los médicos ya usan en la cabecera del paciente.

    No son features nuevas sino combinaciones conocidas de las existentes. Se incluyen por dos
    razones: dan al modelo relaciones no lineales que tendría que redescubrir solo (un cociente
    entre dos columnas es difícil de aprender para un árbol), y hacen el resultado explicable
    ante un clínico, que reconoce estos términos.
    """
    u = df_ult
    salida = {}

    def _criterio(condicion: pd.Series, base: pd.Series) -> pd.Series:
        """Convierte una condición booleana a float, propagando NaN si falta el dato base.

        `(NaN > 38)` evalúa a `False` en pandas, no a NaN: sin este ajuste, un paciente sin
        temperatura registrada contaría silenciosamente como "sin fiebre" en vez de "se
        desconoce". Eso sesgaría los puntajes hacia abajo justo en los pacientes con menos
        datos, y contradice la política de imputación del notebook 04 (dejar el nulo real,
        no inventar un valor "seguro" en su lugar).
        """
        return condicion.astype("float32").where(base.notna())

    # Índice de shock: FC/PAS. Por encima de 0,9 sugiere compromiso circulatorio.
    salida["indice_shock"] = u["HR_ult"] / u["SBP_ult"].replace(0, np.nan)

    # Criterios SIRS, la definición clásica de respuesta inflamatoria sistémica
    sirs_temp = _criterio((u["Temp_ult"] > 38) | (u["Temp_ult"] < 36), u["Temp_ult"])
    sirs_fc = _criterio(u["HR_ult"] > 90, u["HR_ult"])
    sirs_resp = _criterio(u["Resp_ult"] > 20, u["Resp_ult"])
    sirs_leuco = _criterio((u["WBC_ult"] > 12) | (u["WBC_ult"] < 4), u["WBC_ult"])
    salida["sirs_total"] = sirs_temp + sirs_fc + sirs_resp + sirs_leuco  # NaN si falta cualquiera de las 4

    # qSOFA: tamizaje rápido de sepsis fuera de UCI (2 de 3 criterios = alto riesgo).
    # Sin escala de Glasgow en el dataset, se calcula la versión de dos criterios.
    salida["qsofa_parcial"] = (_criterio(u["SBP_ult"] <= 100, u["SBP_ult"])
                               + _criterio(u["Resp_ult"] >= 22, u["Resp_ult"]))

    # Componentes de SOFA disponibles: relación PaO2/FiO2 no es calculable (no hay PaO2),
    # pero sí los marcadores de disfunción renal, hepática, hematológica y circulatoria.
    salida["hipotension"] = _criterio(u["MAP_ult"] < 65, u["MAP_ult"])
    salida["ratio_bun_creat"] = u["BUN_ult"] / u["Creatinine_ult"].replace(0, np.nan)
    salida["presion_pulso"] = u["SBP_ult"] - u["DBP_ult"]

    return pd.DataFrame(salida, index=df_ult.index)


def construir_matriz(df: pd.DataFrame, cfg: dict, cols: list[str] | None = None) -> pd.DataFrame:
    """Orquesta las cuatro familias de features y devuelve la matriz lista para modelar."""
    cols = cols or CLINICAS
    f = cfg["features"]
    df = ordenar(df)

    ult = valor_arrastrado(df, cols, f["max_horas_ffill"])
    bloques = [
        df[["pid", "hosp", "ICULOS", "SepsisLabel"] + cfg["datos"]["cols_demograficas"]],
        ult,
        indicadores_medicion(df, cols),
        carga_de_monitoreo(df, f["ventanas_horas"]),
        dinamica_en_ventana(ult, df["pid"], VITALES, f["ventanas_horas"]),
        desviacion_del_basal(ult, df["pid"], VITALES, f["horas_basal"]),
        scores_clinicos(ult),
    ]

    matriz = pd.concat(bloques, axis=1)

    # Horas transcurridas desde el ingreso al hospital, no solo a la UCI: distingue al paciente
    # que llega directo a UCI del que se deterioró tras varios días en hospitalización.
    matriz["horas_en_hospital"] = matriz["ICULOS"] - matriz["HospAdmTime"]

    numericas = matriz.select_dtypes(include=["float64"]).columns
    matriz[numericas] = matriz[numericas].astype("float32")

    return matriz
