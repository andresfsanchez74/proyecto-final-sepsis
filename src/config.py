"""Acceso a la configuración y a las rutas del proyecto.

Los notebooks viven en notebooks/ y el resto de carpetas cuelgan de la raíz, así que
resolver rutas relativas desde una celda es frágil. Aquí se localiza la raíz una sola vez
(subiendo hasta encontrar config/config.yaml) y todas las rutas se derivan de ella:
el proyecto funciona igual clonado en cualquier máquina, que es lo que pide la rúbrica
de reproducibilidad.
"""

from pathlib import Path

import yaml


def raiz_proyecto(desde: str | Path | None = None) -> Path:
    """Sube por el árbol de directorios hasta encontrar config/config.yaml."""
    actual = Path(desde or Path.cwd()).resolve()
    for candidato in [actual, *actual.parents]:
        if (candidato / "config" / "config.yaml").exists():
            return candidato
    raise FileNotFoundError(
        "No se encontró config/config.yaml. Ejecuta desde dentro del proyecto."
    )


def cargar_config(desde: str | Path | None = None) -> dict:
    """Devuelve el config.yaml como diccionario, con las rutas ya resueltas a absolutas."""
    raiz = raiz_proyecto(desde)
    with open(raiz / "config" / "config.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    cfg["raiz"] = raiz
    cfg["rutas"] = {nombre: raiz / ruta for nombre, ruta in cfg["rutas"].items()}
    return cfg
