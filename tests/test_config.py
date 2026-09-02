"""Tests de src/config.py: la resolución de rutas que hace reproducible el proyecto
independientemente de dónde se clone (ver docstring del módulo)."""

from config import cargar_config


def test_cargar_config_resuelve_rutas_absolutas():
    cfg = cargar_config()

    assert cfg["raiz"].is_absolute()
    assert (cfg["raiz"] / "config" / "config.yaml").exists()
    for ruta in cfg["rutas"].values():
        assert ruta.is_absolute()
        assert cfg["raiz"] in ruta.parents


def test_cargar_config_incluye_parametros_del_utility_score():
    cfg = cargar_config()

    claves_esperadas = {"dt_early", "dt_optimal", "dt_late", "max_u_tp", "min_u_fn", "u_fp", "u_tn"}
    assert claves_esperadas <= cfg["utility"].keys()
