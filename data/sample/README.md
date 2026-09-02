# data/sample/

Muestra versionada de **50 pacientes** (23 + 2 sépticos de cada hospital — 8% de tasa
séptica, cercana a la prevalencia real de 7,27% de pacientes) para que cualquiera que clone
el repo pueda correr el pipeline sin descargar los ~322 MB del dataset completo (sección 5.3
del [README principal](../../README.md)). La selección está en [`manifest.csv`](manifest.csv)
y se generó con semilla fija (42), agrupando por paciente en `data/interim/cohorte_horaria.parquet`.

Mismo layout que `data/raw/` (`sepsis_dataset/training_setX/training_setX/*.psv`), así que
apuntar `rutas.raw` a esta carpeta en `config/config.yaml` corre notebooks/tests sin editar
código:

```python
cfg["rutas"]["raw"] = cfg["raiz"] / "data" / "sample" / "sepsis_dataset"
```

**No reemplaza al dataset completo**: sirve para verificar que el pipeline corre de punta a
punta (rutas, tipos, features) — los resultados y métricas reportados en el README y los
notebooks usan siempre los 40.336 pacientes completos.
