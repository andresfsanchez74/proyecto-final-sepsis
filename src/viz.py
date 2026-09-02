"""Estilo y gráficos que se repiten en varios notebooks.

Solo entran aquí las figuras que se usan más de una vez y que deben verse idénticas en todos
los notebooks y en la presentación final. Un gráfico puntual se escribe en su notebook: sacarlo
a un módulo obligaría al lector a saltar de archivo para entender una figura que solo aparece
una vez.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

# Paleta con significado fijo en todo el proyecto: el lector aprende el código de color una vez.
COLOR_SEPSIS = "#c0392b"     # pacientes que desarrollan sepsis
COLOR_CONTROL = "#2c7fb8"    # pacientes que no
COLOR_A = "#7b3294"          # hospital A
COLOR_B = "#008837"          # hospital B


def aplicar_estilo():
    """Estilo común a las ~50 figuras del proyecto: mismo aspecto en los 10 notebooks y en la
    presentación final.

    La base es `SciencePlots` (tipografía y proporciones de figura de publicación científica,
    en vez del `whitegrid` por defecto de seaborn que comparten la mayoría de proyectos del
    curso). Se usa `no-latex` a propósito: así cualquiera del equipo reproduce las mismas
    figuras sin necesitar una instalación de LaTeX. Encima de esa base se fija la paleta
    semántica del proyecto (`COLOR_SEPSIS`, `COLOR_CONTROL`, `COLOR_A`, `COLOR_B`) — el
    estilo da la tipografía, la paleta da el significado.
    """
    try:
        plt.style.use(["science", "no-latex", "grid"])
    except OSError:
        # Si el estilo no está disponible en el entorno, el notebook sigue funcionando con el
        # estilo por defecto de matplotlib en vez de fallar.
        sns.set_theme(style="whitegrid", context="notebook")

    sns.set_context("notebook")
    plt.rcParams.update({
        "figure.figsize": (10, 5),
        "figure.dpi": 100,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.grid": True,
        "grid.alpha": 0.3,
    })


def guardar(fig, nombre, cfg):
    """Guarda la figura en reports/figures para reutilizarla en la presentación."""
    destino = cfg["rutas"]["figures"] / f"{nombre}.png"
    destino.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destino, dpi=150, bbox_inches="tight")
    return destino


def barras_nulos(porcentajes, titulo, ax=None):
    """Porcentaje de nulos por variable, ordenado, para localizar de un vistazo el problema."""
    ax = ax or plt.subplots(figsize=(9, 10))[1]
    datos = porcentajes.sort_values()
    colores = ["#d73027" if v > 90 else "#fc8d59" if v > 50 else "#91bfdb" for v in datos]

    ax.barh(datos.index, datos.values, color=colores)
    ax.set_xlabel("% de filas-hora sin medición")
    ax.set_title(titulo)
    ax.set_xlim(0, 100)
    return ax


def trayectoria_alineada(df, variable, ax=None, ventana=(-24, 12)):
    """Media de una variable en función de las horas hasta el inicio de sepsis.

    Alinear a todos los pacientes en el instante del evento (en vez de en su hora de ingreso)
    es lo que permite ver si una variable se mueve *antes* de que la sepsis se declare. Con el
    eje en horas de estancia esa señal quedaría difuminada, porque cada paciente enferma en un
    momento distinto de su estancia.
    """
    ax = ax or plt.subplots()[1]
    d = df[df["horas_hasta_sepsis"].between(*ventana)]

    resumen = d.groupby("horas_hasta_sepsis", observed=True)[variable].agg(["mean", "sem"])
    ax.plot(resumen.index, resumen["mean"], color=COLOR_SEPSIS, lw=2)
    ax.fill_between(resumen.index,
                    resumen["mean"] - 1.96 * resumen["sem"],
                    resumen["mean"] + 1.96 * resumen["sem"],
                    color=COLOR_SEPSIS, alpha=0.2)

    ax.axvline(0, color="black", ls="--", lw=1)
    ax.set_xlabel("horas hasta el inicio de sepsis")
    ax.set_ylabel(variable)
    return ax


def curva_entrenamiento(valores, mejor_iteracion=None, metrica="AUPRC validación",
                         titulo="Curva de entrenamiento", ax=None):
    """Evolución de una métrica de validación ronda a ronda de boosting, marcando dónde se
    activó el early stopping.

    Sin esto, un AUPRC alto no distingue un modelo que aprendió de uno que ya empezó a
    sobreajustar antes de que el early stopping lo detuviera: esta curva es la evidencia visual
    de que se controló, no solo se afirmó.
    """
    valores = list(valores)
    ax = ax or plt.subplots(figsize=(8, 5))[1]

    ax.plot(range(1, len(valores) + 1), valores, color=COLOR_CONTROL)
    if mejor_iteracion is not None:
        ax.axvline(mejor_iteracion, color=COLOR_SEPSIS, ls="--",
                   label=f"Mejor iteración ({mejor_iteracion})")
        ax.legend()

    ax.set_xlabel("Ronda de boosting (número de árboles)")
    ax.set_ylabel(metrica)
    ax.set_title(titulo)
    return ax


def distribucion_probabilidades(modelos, X, y, get_proba):
    """Histograma de probabilidades predichas, separado por clase real, un panel por modelo.

    Es el gráfico que explica de un vistazo por qué el umbral 0,5 no tiene ningún significado
    clínico aquí: si las dos distribuciones se solapan tanto como se solapan, cualquier corte
    fijo va a dejar del lado equivocado a una fracción grande de una de las dos clases.
    """
    n = len(modelos)
    fig, axes = plt.subplots(1, n, figsize=(5.5 * n, 4.5), sharey=True)
    axes = np.atleast_1d(axes)

    for ax, (nombre, modelo) in zip(axes, modelos.items()):
        prob = get_proba(modelo, X)
        sns.histplot(x=prob[y == 0], bins=50, stat="density", color=COLOR_CONTROL,
                     label="Real: no sepsis", alpha=0.5, ax=ax)
        sns.histplot(x=prob[y == 1], bins=50, stat="density", color=COLOR_SEPSIS,
                     label="Real: sepsis", alpha=0.5, ax=ax)
        ax.set_xlabel("Probabilidad predicha de sepsis")
        ax.set_title(nombre)
        ax.legend()

    plt.tight_layout()
    return fig


def matrices_confusion(modelos, X, y, get_proba, umbral=0.5):
    """Matriz de confusión de cada modelo, en un panel, al umbral de decisión indicado.

    El panel comparativo importa tanto como el número: pone a los cinco modelos a repartirse
    los mismos falsos positivos y falsos negativos bajo la misma regla de decisión, en vez de
    juzgar cada uno con una matriz suelta que nadie puede comparar de un vistazo.
    """
    n = len(modelos)
    fig, axes = plt.subplots(1, n, figsize=(4.6 * n, 4.2))
    axes = np.atleast_1d(axes)

    for ax, (nombre, modelo) in zip(axes, modelos.items()):
        prob = get_proba(modelo, X)
        pred = (prob >= umbral).astype(int)
        cm = confusion_matrix(y, pred)
        sns.heatmap(
            cm, annot=True, fmt=",d", cmap="Blues", ax=ax, cbar=False,
            xticklabels=["No sepsis", "Sepsis"], yticklabels=["No sepsis", "Sepsis"],
        )
        ax.set_xlabel("Predicción")
        ax.set_ylabel("Real")
        ax.set_title(f"{nombre}\n(umbral={umbral})")

    plt.tight_layout()
    return fig


def precision_recall_f1_vs_umbral(y_true, y_score, umbral_elegido=None, ax=None):
    """Precisión, recall y F1 en función del umbral, para un solo modelo, sin reentrenarlo.

    Es la lectura "clásica" del umbral de decisión — la que se espera ver antes de justificar
    por qué el proyecto elige el suyo con el utility score en vez de con F1 (notebook 07): F1
    pesa igual un falso positivo que un falso negativo, y en este problema esos dos errores no
    cuestan lo mismo ni de lejos.
    """
    umbrales = np.linspace(0.02, 0.95, 60)
    y_true = np.asarray(y_true)
    filas = []
    for u in umbrales:
        pred = (y_score >= u).astype(int)
        filas.append({
            "umbral": u,
            "precision": precision_score(y_true, pred, zero_division=0),
            "recall": recall_score(y_true, pred, zero_division=0),
            "f1": f1_score(y_true, pred, zero_division=0),
        })
    curva = pd.DataFrame(filas)

    ax = ax or plt.subplots(figsize=(9, 5.5))[1]
    ax.plot(curva["umbral"], curva["precision"], label="Precisión", color=COLOR_A)
    ax.plot(curva["umbral"], curva["recall"], label="Recall", color=COLOR_SEPSIS)
    ax.plot(curva["umbral"], curva["f1"], label="F1", color=COLOR_CONTROL)
    if umbral_elegido is not None:
        ax.axvline(umbral_elegido, color="black", ls="--", lw=1,
                   label=f"Umbral elegido (utility, {umbral_elegido:.3f})")
    ax.set_xlabel("Umbral de decisión")
    ax.set_ylabel("Valor de la métrica")
    ax.set_title("Precisión, recall y F1 según el umbral")
    ax.legend()
    return ax
