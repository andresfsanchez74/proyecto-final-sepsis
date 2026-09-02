# Predicción Temprana de Sepsis en UCI

**Curso:** Máquina de Aprendizaje 1 — Universidad de La Sabana
**Profesor:** Jesús Antonio Villarraga P.
**Autores:** Daniel Forero, Cristian Manuel Castañeda Gutiérrez
**Metodología:** CRISP-DM

---

## 1. Descripción del problema

La sepsis es una de las principales causas de muerte en Unidades de Cuidados Intensivos (UCI):
según la Organización Mundial de la Salud, se estima que desarrolla sepsis cerca de 30 millones
de personas al año en el mundo y mueren por su causa unas 6 millones; solo en Estados Unidos
supera los 24.000 millones de dólares anuales en costos de atención. Su tratamiento
(antibióticos, fluidos) es sensible al tiempo: cada hora de retraso se asocia a un 4-8% más de
mortalidad. En la práctica clínica, la sepsis suele reconocerse cuando el deterioro ya es
evidente, lo que limita la ventana de intervención efectiva.

Este proyecto usa el dataset del **PhysioNet / Computing in Cardiology Challenge 2019**, que
contiene registros horarios de signos vitales, resultados de laboratorio y datos demográficos
de pacientes de UCI de dos hospitales, con una etiqueta binaria (`SepsisLabel`) que indica si
el paciente cumple los criterios clínicos Sepsis-3 en cada hora de su estancia.

## 2. Objetivos

**Objetivo general:** desarrollar un modelo de clasificación capaz de anticipar el desarrollo
de sepsis en pacientes de UCI a partir de sus variables clínicas horarias, como apoyo a un
sistema de alerta temprana.

**Objetivos específicos:**
1. Comprender la estructura, calidad y comportamiento de los datos clínicos disponibles.
2. Preparar los datos (limpieza, imputación validada con evidencia, ingeniería de variables,
   partición por paciente) para el modelado.
3. Entrenar y comparar modelos de clasificación adecuados para un problema fuertemente
   desbalanceado y con estructura de serie temporal.
4. Evaluar los modelos con el utility score oficial del challenge — no solo con AUROC/AUPRC —
   e interpretar los resultados en términos clínicos.
5. Comunicar los hallazgos de forma clara, incluyendo en qué condiciones el modelo rinde mejor
   y cuáles son sus límites reales.

## 3. Tipo de problema

**Clasificación binaria supervisada**, evaluada hora a hora por paciente:
- Clase `0`: el paciente no cumple criterios de sepsis en esa hora.
- Clase `1`: el paciente cumple criterios de sepsis en esa hora.

Dos retos centrales condicionan todas las decisiones metodológicas: **fuerte desbalance de
clases** (1,80% de las horas son positivas) y **estructura de serie temporal por paciente**
(cada paciente aporta una secuencia de horas, no observaciones independientes).

## 4. Datos

- **Fuente:** [PhysioNet Computing in Cardiology Challenge 2019](https://physionet.org/content/challenge-2019/1.0.0/) — *Training Set A* y *Training Set B* completos.
- **Formato:** un archivo `.psv` (valores separados por `|`) por paciente; cada fila es una
  hora de estancia en UCI. El identificador de paciente y el hospital no vienen en el archivo:
  se rescatan del nombre y de la carpeta al cargar (notebook 00).
- **Tamaño:** 40.336 pacientes (20.336 del hospital A, 20.000 del B) · 1.552.210 registros
  hora-paciente · 40 variables predictoras + 1 objetivo.
- **Variables:** 8 signos vitales, 26 resultados de laboratorio, 6 variables
  demográficas/administrativas. El diccionario de datos completo está en
  [`notebooks/01_exploracion_cohorte.ipynb`](notebooks/01_exploracion_cohorte.ipynb).
- **Etiquetado:** criterio clínico Sepsis-3 (antibióticos + cultivos de sangre, y un aumento de
  2 puntos en el score SOFA dentro de 24h), documentado en detalle en
  [`notebooks/08_interpretabilidad.ipynb`](notebooks/08_interpretabilidad.ipynb) — la etiqueta
  ya viene desplazada 6h hacia atrás por el organizador del challenge.

El dataset completo **no se versiona en este repositorio** por su tamaño (>300 MB, 40.336
archivos). Ver la sección 5.3 para instrucciones de descarga.

## 5. Instrucciones de ejecución

### 5.1 Requisitos

- Python 3.11+
- Ver [`requirements.txt`](requirements.txt) para la lista completa de librerías.

### 5.2 Instalación

```bash
git clone <URL-del-repositorio>
cd proyecto-final-sepsis

python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux / Mac:
source .venv/bin/activate

pip install -r requirements.txt
```

### 5.3 Obtener los datos

1. Descargar *Training Set A* y *Training Set B* desde PhysioNet:
   https://physionet.org/content/challenge-2019/1.0.0/
2. Descomprimir de forma que los archivos `.psv` queden en:
   ```
   data/raw/sepsis_dataset/training_setA/training_setA/p000001.psv
   data/raw/sepsis_dataset/training_setB/training_setB/p100001.psv
   ...
   ```
   (la ruta exacta es configurable en [`config/config.yaml`](config/config.yaml), sección `datos.hospitales`).

3. Alternativa rápida sin descargar los 322 MB: [`data/sample/`](data/sample/README.md) trae
   50 pacientes ya versionados en el repo, con la misma estructura de carpetas — apuntar
   `rutas.raw` ahí corre el pipeline completo (notebooks y tests) para verificar que todo
   funciona, aunque los resultados reportados en este README usan siempre el dataset completo.

### 5.4 Ejecutar el proyecto

```bash
jupyter notebook
```

Ejecutar los notebooks de la carpeta `notebooks/` **en orden numérico** (cada uno corresponde a
una fase de CRISP-DM; ver la tabla de la sección 6). La configuración de rutas, semilla y
parámetros del proyecto vive en [`config/config.yaml`](config/config.yaml) y la lee
automáticamente el código en `src/` a través de `src/config.py` — no hace falta editar rutas a
mano en ningún notebook, el proyecto se resuelve solo desde donde se clone.

Los notebooks 06, 06b, 07 y 08 dependen de los artefactos que guardan los anteriores
(`data/processed/matriz_features_completa.parquet`, `models/*.joblib`, `models/*.pt`,
`models/*.csv`) — si se ejecutan fuera de orden, van a fallar al no encontrarlos.

### 5.5 Ejecutar los tests

```bash
pytest tests/
```

Cubren `src/utility.py`, `src/evaluacion.py` y `src/config.py` con datos sintéticos — no
necesitan el dataset descargado, así que pueden correr justo después de `pip install -r
requirements.txt`.

## 6. Metodología (CRISP-DM)

| Fase CRISP-DM | Notebook(s) | Estado |
|---|---|---|
| 1. Comprensión del negocio | Introducción de `01_exploracion_cohorte.ipynb` y este README | ✅ Completo |
| 2. Comprensión de los datos | `00`, `01`, `02`, `03` | ✅ Completo |
| 3. Preparación de los datos | `04`, `05` | ✅ Completo |
| 4. Modelado | `06`, `06b` | ✅ Completo |
| 5. Evaluación | `07` | ✅ Completo |
| 6. Interpretación de resultados | `08` | ✅ Completo |

### 6.1 Comprensión de los datos (`00`–`03`)

- **`00_carga_consolidacion`**: convierte los 40.336 `.psv` en una sola tabla auditada —
  verifica que el reloj (`ICULOS`) esté completo y ordenado, que la etiqueta sea monótona, y
  corrige 356 mediciones fisiológicamente imposibles (solo la celda, nunca la fila).
- **`01_exploracion_cohorte`**: quiénes son estos pacientes — dos escalas del objetivo (1,80%
  de horas, 7,27% de pacientes), demografía, duración de estancia, y el hallazgo que atraviesa
  todo el proyecto: los hospitales A y B son **dos sistemas clínicos y administrativos
  distintos**, no dos muestras del mismo proceso.
- **`02_analisis_nulos_y_tiempos`**: la ausencia de un dato es informativa — lactato y FiO₂ se
  piden ~2,3 veces más en pacientes que terminan sépticos.
- **`03_exploracion_temporal_sepsis`**: trayectorias individuales, perfiles multivariantes y
  líneas de tiempo — confirma que el problema debe resolverse hora a hora, sin atajos.

### 6.2 Preparación de datos (`04`–`05`)

- **`04_imputacion_estrategias`**: la política de imputación **se valida con un experimento**
  (enmascarar valores reales y medir el error de recuperarlos), no se asume. Resultado: arrastre
  causal (último valor conocido, tope de 24h) para todas las variables, con indicador de
  medición y antigüedad para los laboratorios dispersos — nunca imputación global, que deforma
  la distribución real.
- **`05_feature_engineering`**: 40 columnas crudas → 209 features, en cuatro familias con
  hipótesis clínica propia (valor arrastrado, indicadores de medición, dinámica en ventanas de
  6h/24h, desviación del basal propio del paciente) más scores clínicos (SIRS, qSOFA, índice de
  shock). Partición final por paciente (nunca por fila), 80/20.

### 6.3 Modelado (`06`, `06b`)

Cinco modelos tabulares bajo las mismas reglas — regresión logística, HistGradientBoosting,
LightGBM, XGBoost y CatBoost — más un GRU secuencial que ve el orden real de las horas.

**Resultados en el escenario intra-cohorte** (mismo test, ordenados por AUPRC):

| Modelo | AUROC | AUPRC | Lift sobre prevalencia |
|---|---|---|---|
| **CatBoost** (campeón) | **0,8497** | **0,1229** | 7,28× |
| XGBoost | 0,8429 | 0,1127 | 6,68× |
| HistGradientBoosting | 0,8375 | 0,1144 | 6,78× |
| LightGBM | 0,8281 | 0,1032 | 6,12× |
| GRU (secuencial) | 0,8165 | 0,1033 | 6,12× |
| Regresión logística | 0,8037 | 0,0869 | 5,15× |

`CatBoost` gana en las dos métricas a la vez y es, de los cinco, el más consistente al validar
cruzando de hospital (entrenar en uno, evaluar en el otro) — la caída de desempeño al cruzar
(AUROC de ~0,85 a ~0,68-0,75) confirma con modelos, no solo con estadística, que A y B son
sistemas distintos. Se probó también si especializar un modelo por hospital mejora el
resultado: para el campeón, no — el modelo mixto gana en los dos hospitales evaluados.

### 6.4 Evaluación (`07`)

El utility score oficial del challenge (verificado contra `evaluate_sepsis_score.py` del
organizador, no de memoria) fija el umbral de decisión con una regla que **nunca mira las
etiquetas** — alarmar solo en el 5% de horas con score más alto, el presupuesto real que una
UCI toleraría — en vez de maximizar utility sobre el propio test, que sería fuga de
información.

**Resultado del campeón:** utility normalizado de **0,3575**, detectando al 43,0% de los
sépticos del test con una mediana de 45 horas de anticipación. Comparado contra el leaderboard
oficial del challenge (Reyna et al. 2019), el resultado queda en el mismo orden de magnitud que
el top-5 mundial en hospitales del mismo tipo (0,40-0,43) — y el propio equipo ganador del
challenge saca utility **negativo** en un hospital nunca visto, la misma limitación de
generalización entre hospitales que este proyecto documenta con evidencia propia.

### 6.5 Interpretabilidad (`08`)

SHAP sobre el modelo campeón muestra que la señal fisiológica existe y apunta en la dirección
clínica correcta (fiebre, taquipnea, leucocitosis y deterioro renal suben el riesgo), pero que
**el contexto administrativo** (unidad, tiempos de ingreso) está sobre-representado más de 7
veces respecto a su proporción de columnas — la causa más probable de la caída de desempeño al
cruzar de hospital. Se documenta también, con datos propios, que la sepsis es un problema
genuinamente difícil de predecir: la definición clínica (Sepsis-3) depende de un score de 6
sistemas de órganos del que este dataset no puede reconstruir 2 por completo (sistema nervioso,
mitad del respiratorio), y las variables disponibles se solapan sustancialmente entre pacientes
sépticos y no sépticos.

## 7. Estructura del repositorio

```
├── config/                 # Configuración central (rutas, semilla, parámetros del utility score)
│   └── config.yaml
├── data/
│   ├── README.md              # Qué carpetas hay y cómo poblarlas (ver sección 5.3)
│   ├── raw/                 # Dataset completo (NO versionado; ver sección 5.3)
│   ├── sample/                # 50 pacientes SÍ versionados, para correr el pipeline sin descargar todo
│   ├── interim/              # Cohorte consolidada (no versionado, se regenera con el notebook 00)
│   └── processed/            # Matriz de features lista para modelar (no versionado, notebook 05)
├── models/                 # Modelos entrenados y tablas de resultados (no versionado salvo los CSV)
├── notebooks/               # Un notebook por fase de CRISP-DM, ejecutados en orden
│   ├── 00_carga_consolidacion.ipynb
│   ├── 01_exploracion_cohorte.ipynb
│   ├── 02_analisis_nulos_y_tiempos.ipynb
│   ├── 03_exploracion_temporal_sepsis.ipynb
│   ├── 04_imputacion_estrategias.ipynb
│   ├── 05_feature_engineering.ipynb
│   ├── 06_modelado_baselines.ipynb        # 5 modelos tabulares + validación cruzada entre hospitales
│   ├── 06b_modelado_secuencial.ipynb      # GRU
│   ├── 07_evaluacion_utility_umbral.ipynb # Utility score oficial y elección de umbral
│   └── 08_interpretabilidad.ipynb         # SHAP, Yellowbrick, y por qué sepsis es difícil de predecir
├── reports/
│   ├── figures/              # Gráficos exportados (viz.guardar(), ver notebooks)
│   ├── dashboard/            # Insumos para el dashboard: importancia_shap.csv, test_scored.parquet
│   └── tables/                # Tablas comparativas exportadas
├── src/                     # Código reutilizable importado por los notebooks
│   ├── config.py             # Carga de config.yaml y resolución de rutas absolutas
│   ├── io_datos.py           # Lectura y consolidación de los archivos .psv
│   ├── features.py           # Ingeniería de características, estrictamente causal
│   ├── evaluacion.py         # Particiones sin fuga y métricas para un problema desbalanceado
│   ├── utility.py            # Reimplementación del utility score oficial del challenge
│   ├── viz.py                # Estilo visual único del proyecto y gráficos reutilizados
│   └── serving.py            # Predicción sobre datos nuevos, para el simulador del dashboard
├── tests/                   # Tests de src/ con datos sintéticos (pytest tests/, ver 5.5)
│   ├── test_utility.py
│   ├── test_evaluacion.py
│   ├── test_config.py
│   └── test_serving.py        # Se salta si no hay modelo entrenado localmente (models/*.joblib)
├── requirements.txt
├── .gitignore
└── README.md
```

## 8. Resultados principales

- **Modelo campeón:** CatBoost, utility normalizado 0,3575, umbral de decisión 0,7838 —
  detecta al 43,0% de los pacientes sépticos del test con una mediana de 45 horas de
  anticipación, dentro de un presupuesto de alarma del 5% de las horas.
- **La validación cruzada entre hospitales confirma, con modelos entrenados, que los
  hospitales A y B son sistemas clínicos distintos**: todo modelo pierde desempeño real al
  cruzar (AUROC de ~0,85 a ~0,68-0,75), y ese mismo patrón aparece en el leaderboard oficial
  del challenge — los cinco mejores equipos del mundo sacan utility negativo en un hospital que
  nunca vieron.
- **"Mejor" depende de qué se mida**: como clasificador (AUROC/AUPRC) el campeón rinde mejor en
  el hospital B; como sistema de alarma ya desplegado con un umbral fijo, rinde mejor en A —
  por cómo interactúa un umbral calibrado globalmente con la distribución de scores de cada
  hospital.
- **Especializar un modelo por hospital no mejora al campeón** en ninguno de los dos hospitales
  evaluados — un modelo mixto sigue siendo la mejor opción de despliegue.
- **SHAP muestra que casi una cuarta parte de la señal del modelo viene de contexto
  administrativo** (unidad, tiempos de ingreso), no de fisiología — la explicación más probable
  de por qué el modelo no generaliza mejor entre hospitales.

## 9. Limitaciones

- **Precisión baja en el punto de operación clínico**: con 1,69% de prevalencia real, el mejor
  modelo genera cerca de 6,8 falsas alarmas por cada acierto al presupuesto de alarma del 5%.
  Es el costo estructural de un problema fuertemente desbalanceado, no una falla de diseño.
- **Dependencia de contexto administrativo**: una parte relevante del desempeño del modelo se
  apoya en variables que no viajan a otro hospital (unidad, tiempos de ingreso), lo que limita
  la promesa de generalización a centros no vistos durante el entrenamiento.
- **El dataset no puede reconstruir la definición clínica completa de sepsis**: de los 6
  sistemas de órganos del score SOFA que en parte definen la etiqueta, faltan por completo la
  escala de Glasgow (sistema nervioso) y la mitad del criterio respiratorio (no hay PaO₂), y
  falta la dosis de vasopresores del criterio cardiovascular — un techo de predictibilidad
  estructural, no solo de modelado.
- **El modelo puede ignorar un valor de laboratorio anormal aislado** si llega acompañado de
  poca intensidad de vigilancia clínica alrededor (documentado con un caso real en el
  notebook 08) — aprendió que la frecuencia de medición es señal real, y eso a veces juega en
  contra.
- **Etiqueta truncada tras el evento**: el registro de casi todos los pacientes sépticos se
  corta ~9 horas después del inicio de sepsis, lo que limita cualquier análisis de trayectorias
  más allá de esa ventana.

## 10. Uso de herramientas de IA

Se utilizó una herramienta de IA (Claude Code) como apoyo para la escritura inicial de código
exploratorio, la fusión de dos versiones independientes del proyecto en una sola, y la
redacción de documentación y estructura del repositorio. Todo el código y las interpretaciones
fueron revisados, ejecutados de extremo a extremo (verificando ausencia de errores) y validados
por el equipo contra los resultados numéricos reales del dataset antes de incluirse en este
repositorio. Las decisiones metodológicas (qué modelos usar, qué métricas priorizar, cómo tratar
el desbalance de clases y los datos faltantes, cómo interpretar los hallazgos clínicos) son
criterio del equipo, justificado técnicamente en cada notebook con evidencia propia — cada
afirmación sobre los datos está respaldada por una celda ejecutable que la sustenta, no por una
generalización citada de memoria.
