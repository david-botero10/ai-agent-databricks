# Databricks notebook source
# MAGIC %md
# MAGIC # Parte 5 — ¿Se puede confiar en el juez?
# MAGIC
# MAGIC Un juez LLM también se equivoca. Este notebook:
# MAGIC
# MAGIC 1. Toma **10 respuestas** de la última evaluación, de distintos tipos de pregunta.
# MAGIC 2. Las exporta para que una persona las califique a mano (sin ver el veredicto del juez).
# MAGIC 3. Calcula el **acuerdo juez–humano** y lo deja registrado en la traza.

# COMMAND ----------

# MAGIC %run ../_resources/01-setup

# COMMAND ----------

import os

import mlflow
import pandas as pd

xp_name = os.getcwd().rsplit("/", 1)[0] + "/02-agent-eval/02.1_agent_evaluation"
mlflow.set_experiment(xp_name)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1 · Tomar las 10 respuestas de la última evaluación

# COMMAND ----------

EVAL_XP = xp_name

runs = mlflow.search_runs(
    experiment_names=[EVAL_XP],
    order_by=["attributes.start_time DESC"],
    max_results=1,
)

if runs.empty:
    raise Exception("No hay evaluaciones registradas todavía. Ejecutá primero el notebook 02.1.")

run_id = runs.iloc[0]["run_id"]
print("Run de evaluación:", run_id, runs.iloc[0]["tags.mlflow.runName"])

tabla = mlflow.load_table(
    "eval_results",
    run_ids=[run_id],
    extra_columns=["trace"],
)

print(f"{len(tabla)} filas")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2 · Seleccionar 10 respuestas variadas
# MAGIC Se intenta cubrir los cuatro tipos; si algún tipo no llega al mínimo, se completa con las que haya.

# COMMAND ----------

tabla["question_type"] = tabla["expectations"].apply(
    lambda e: (e or {}).get("question_type", "desconocido")
)

seleccion = []
for tipo in ["directa", "multi_documento", "sin_respuesta", "adversaria"]:
    subset = tabla[tabla["question_type"] == tipo].head(3)
    seleccion.append(subset)
muestra = pd.concat(seleccion).head(10)

print(muestra["question_type"].value_counts())

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3 · Hoja de etiquetado
# MAGIC `humano_correcto` queda vacío: lo completa una persona. **No mires `correctness` antes de calificar.**

# COMMAND ----------

hoja = pd.DataFrame(
    {
        "pregunta": muestra["inputs"].apply(lambda i: (i or {}).get("question", "")),
        "tipo": muestra["question_type"],
        "respuesta_esperada": muestra["expectations"].apply(lambda e: (e or {}).get("expected_response", "")),
        "respuesta_del_bot": muestra["outputs"],
        "humano_correcto": "",
        "trace_id": muestra["trace_id"],
    }
)

ruta = f"/Volumes/{catalog}/{dbName}/{volume_name}/etiquetado_humano.csv"
hoja.to_csv(ruta, index=False)
print("Hoja guardada en", ruta)

display(hoja[["pregunta", "tipo", "respuesta_del_bot"]])

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4 · Calcular el acuerdo juez–humano
# MAGIC Se ejecuta **después** de completar la columna `humano_correcto` con `si` / `no`
# MAGIC (o `true` / `false`) y volver a leer el CSV.

# COMMAND ----------

VALORES_TRUE = {"si", "sí", "true", "1", "yes"}
VALORES_FALSE = {"no", "false", "0"}

etiquetado = spark.createDataFrame(pd.read_csv(ruta)).toPandas()
etiquetado = etiquetado[etiquetado["humano_correcto"].astype(str).str.strip() != ""]

if len(etiquetado) == 0:
    print("⚠️ Todavía no hay etiquetas humanas. Completá el CSV y volvé a ejecutar esta celda.")
else:
    humano = etiquetado["humano_correcto"].astype(str).str.strip().str.lower().isin(VALORES_TRUE)

    jueces = {}
    for col in tabla.columns:
        if col.startswith("correctness/") and col.endswith("/score"):
            jueces[col] = tabla.set_index("trace_id")[col]

    filas = []
    for nombre, serie in jueces.items():
        juez = etiquetado["trace_id"].map(serie)
        valido = juez.notna()
        acuerdo = (juez[valido].astype(bool) == humano[valido]).mean()
        filas.append({"juez": nombre, "acuerdo": round(float(acuerdo), 3), "n": int(valido.sum())})

    resumen = pd.DataFrame(filas)
    display(resumen)

    with mlflow.start_run(run_name="acuerdo_juez_humano"):
        for _, fila in resumen.iterrows():
            mlflow.log_metric(f"acuerdo_{fila['juez']}", fila["acuerdo"])
        mlflow.log_metric("n_etiquetadas", len(etiquetado))
    print("Acuerdo registrado en MLflow")
