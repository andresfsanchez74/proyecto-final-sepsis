"""Consolidación de los 40.336 archivos .psv del challenge en una sola tabla horaria.

El dataset viene como un archivo por paciente, sin identificador dentro del archivo: la
identidad del paciente está solo en el nombre (p000001.psv) y el hospital solo en la carpeta.
Ambos son información imprescindible — el pid para agrupar los splits y evitar fuga, el
hospital para el análisis de transferencia entre sitios — así que se recuperan al cargar.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm


def listar_archivos(cfg) -> list[tuple[Path, str]]:
    """Devuelve [(ruta, hospital)] para todos los pacientes de ambos hospitales."""
    raw = cfg["rutas"]["raw"]
    archivos = []
    for hospital, subcarpeta in cfg["datos"]["hospitales"].items():
        rutas = sorted((raw / subcarpeta).glob("*.psv"))
        if not rutas:
            raise FileNotFoundError(f"No hay .psv en {raw / subcarpeta}")
        archivos += [(r, hospital) for r in rutas]
    return archivos


def cargar_cohorte(cfg, mostrar_progreso: bool = True) -> pd.DataFrame:
    """Lee todos los .psv y los apila en un único DataFrame en formato largo."""
    archivos = listar_archivos(cfg)
    iterador = tqdm(archivos, desc="Leyendo pacientes") if mostrar_progreso else archivos

    trozos = []
    for ruta, hospital in iterador:
        d = pd.read_csv(ruta, sep="|")
        d["pid"] = ruta.stem          # p000001 -> identifica al paciente en los splits
        d["hosp"] = hospital
        trozos.append(d)

    df = pd.concat(trozos, ignore_index=True)

    # float32 basta para mediciones clínicas (2 decimales) y reduce a la mitad la memoria
    # de una tabla de ~1,5M filas; float64 aquí solo gastaría RAM sin ganar precisión.
    numericas = df.select_dtypes(include=[np.number]).columns.drop(["SepsisLabel", "ICULOS"])
    df[numericas] = df[numericas].astype("float32")
    df["SepsisLabel"] = df["SepsisLabel"].astype("int8")
    df["ICULOS"] = df["ICULOS"].astype("int16")
    df["pid"] = df["pid"].astype("category")
    df["hosp"] = df["hosp"].astype("category")

    return df


def guardar_cohorte(df: pd.DataFrame, cfg) -> Path:
    """Persiste la cohorte en parquet (no CSV: 1,5M filas y tipos que hay que preservar)."""
    destino = cfg["rutas"]["interim"] / cfg["datos"]["archivo_cohorte"]
    destino.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(destino, index=False)
    return destino


def leer_cohorte(cfg) -> pd.DataFrame:
    """Lee la cohorte ya consolidada; si no existe, indica qué notebook la genera."""
    origen = cfg["rutas"]["interim"] / cfg["datos"]["archivo_cohorte"]
    if not origen.exists():
        raise FileNotFoundError(
            f"Falta {origen}. Ejecuta primero notebooks/00_carga_consolidacion.ipynb"
        )
    return pd.read_parquet(origen)
