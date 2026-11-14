# Databricks notebook source
# MAGIC %md
# MAGIC # Parte 5 — ¿Se puede confiar en el juez?
# MAGIC
# MAGIC Un juez LLM también se equivoca. Este notebook:
# MAGIC
# MAGIC 1. **`modo=exportar`**: toma 10 respuestas de una evaluación (de los cuatro tipos de pregunta) y genera la hoja
# MAGIC    `etiquetado_humano.csv` para que una persona las califique **sin ver el veredicto del juez**.
# MAGIC 2. **`modo=calcular`**: lee la hoja calificada, registra cada calificación humana en su traza
# MAGIC    (`correctness_humano`, fuente `HUMAN`) y califica las mismas 10 respuestas con **tres jueces** de `Correctness`:
# MAGIC    - el juez gestionado de Databricks,
# MAGIC    - `databricks-gpt-oss-120b` (otra familia de modelos que el agente),
# MAGIC    - `databricks-meta-llama-3-3-70b-instruct` (el mismo modelo del agente: mide el sesgo de autoevaluación).
# MAGIC 3. Calcula el **acuerdo juez–humano** de cada juez y lista los desacuerdos.

# COMMAND ----------

# MAGIC %pip install "mlflow>=3.10.1" databricks-agents "typing_extensions>=4.13"

# COMMAND ----------

# Reinicia Python para cargar las versiones recién instaladas (p. ej. typing_extensions)
dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %run ./_resources/01-setup

# COMMAND ----------

import json
import os

import mlflow
import pandas as pd
from mlflow.entities import AssessmentSource, AssessmentSourceType
from mlflow.genai.scorers import Correctness

dbutils.widgets.dropdown("modo", "calcular", ["exportar", "calcular"])
dbutils.widgets.text("eval_run_name", "exp_base")
dbutils.widgets.text("etiquetador", "david.botero")
MODO = dbutils.widgets.get("modo")
EVAL_RUN_NAME = dbutils.widgets.get("eval_run_name")
ETIQUETADOR = dbutils.widgets.get("etiquetador")

xp_name = os.getcwd() + "/02-agent-eval/02.1_agent_evaluation"
mlflow.set_experiment(xp_name)
RUTA = f"/Volumes/{catalog}/{dbName}/{volume_name}/etiquetado_humano.csv"

JUECES = {
    "juez_gestionado": None,
    "juez_gpt_oss_120b": "databricks:/databricks-gpt-oss-120b",
    "juez_llama_3_3_70b": "databricks:/databricks-meta-llama-3-3-70b-instruct",
}

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1 · Exportar 10 respuestas para calificar

# COMMAND ----------

def question_of(trace):
    return trace.data.spans[0].inputs["request"]["input"][0]["content"]


def answer_of(trace):
    for item in reversed(trace.data.response.get("output", [])):
        if item.get("type") == "message":
            return item["content"][-1]["text"]
    return ""


def expectation(trace, name):
    return next((a.expectation.value for a in trace.info.assessments
                 if getattr(a, "expectation", None) and a.name == name), "")


if MODO == "exportar":
    runs = mlflow.search_runs(filter_string=f"tags.mlflow.runName = '{EVAL_RUN_NAME}'",
                              order_by=["attributes.start_time DESC"], max_results=1)
    assert not runs.empty, f"No existe el run de evaluación {EVAL_RUN_NAME}"
    traces = [t for t in mlflow.search_traces(run_id=runs.iloc[0]["run_id"], return_type="list")
              if t.info.state.name == "OK" and t.data.response]
    por_tipo = {"directa": 3, "multi_documento": 2, "sin_respuesta": 2, "adversaria": 3}
    filas = []
    for tipo, n in por_tipo.items():
        elegidas = [t for t in traces if expectation(t, "question_type") == tipo][:n]
        for t in elegidas:
            filas.append({"pregunta": question_of(t), "tipo": tipo,
                          "respuesta_esperada": expectation(t, "expected_response"),
                          "respuesta_del_bot": answer_of(t), "humano_correcto": "", "trace_id": t.info.trace_id})
    hoja = pd.DataFrame(filas)
    if os.path.exists(RUTA) and pd.read_csv(RUTA)["humano_correcto"].notna().any():
        raise Exception(f"{RUTA} ya tiene calificaciones humanas: no se sobrescribe")
    hoja.to_csv(RUTA, index=False)
    print(f"Hoja con {len(hoja)} respuestas guardada en {RUTA}")
    dbutils.notebook.exit(f"exportadas {len(hoja)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2 · Registrar las calificaciones humanas y calificar con los tres jueces

# COMMAND ----------

VALORES_TRUE = {"si", "sí", "true", "1", "yes"}
VALORES_FALSE = {"no", "false", "0"}

hoja = pd.read_csv(RUTA)
hoja["humano_correcto"] = hoja["humano_correcto"].astype(str).str.strip().str.lower()
validas = hoja["humano_correcto"].isin(VALORES_TRUE | VALORES_FALSE)
assert validas.all(), f"Hay filas sin calificar o con valores no válidos: {hoja.loc[~validas, 'pregunta'].tolist()}"
hoja["humano"] = hoja["humano_correcto"].isin(VALORES_TRUE)

for _, fila in hoja.iterrows():
    mlflow.log_feedback(trace_id=fila["trace_id"], name="correctness_humano", value=bool(fila["humano"]),
                        source=AssessmentSource(source_type=AssessmentSourceType.HUMAN, source_id=ETIQUETADOR))
print(f"{len(hoja)} calificaciones humanas registradas en sus trazas")

datos = [{"inputs": {"question": f["pregunta"]}, "outputs": f["respuesta_del_bot"],
          "expectations": {"expected_response": f["respuesta_esperada"]}} for _, f in hoja.iterrows()]

veredictos = {}
with mlflow.start_run(run_name="acuerdo_juez_humano") as run:
    for nombre, modelo in JUECES.items():
        kwargs = {"model": modelo} if modelo else {}
        res = mlflow.genai.evaluate(data=datos, scorers=[Correctness(name=nombre, **kwargs)])
        df = res.result_df
        col = next(c for c in df.columns if c.startswith(nombre) and c.endswith("value"))
        preguntas = df["request"].apply(lambda r: (r or {}).get("question") if isinstance(r, dict) else json.loads(r)["question"])
        veredictos[nombre] = dict(zip(preguntas, df[col].astype(str).str.lower().isin({"yes", "true"})))

    resumen, desacuerdos = [], []
    for nombre, v in veredictos.items():
        juez = hoja["pregunta"].map(v)
        acuerdo = float((juez == hoja["humano"]).mean())
        resumen.append({"juez": nombre, "acuerdo": round(acuerdo, 2), "aciertos_segun_juez": int(juez.sum())})
        mlflow.log_metric(f"acuerdo_{nombre}", acuerdo)
        for _, f in hoja[juez != hoja["humano"]].iterrows():
            desacuerdos.append({"juez": nombre, "tipo": f["tipo"], "pregunta": f["pregunta"],
                                "humano": f["humano"], "juez_dice": not f["humano"]})
    mlflow.log_metric("aciertos_segun_humano", int(hoja["humano"].sum()))
    mlflow.log_param("eval_run_name", EVAL_RUN_NAME)

resumen = pd.DataFrame(resumen)
desacuerdos = pd.DataFrame(desacuerdos)
resumen.to_csv(RUTA.replace(".csv", "_resumen.csv"), index=False)
desacuerdos.to_csv(RUTA.replace(".csv", "_desacuerdos.csv"), index=False)
display(resumen)
display(desacuerdos)

# COMMAND ----------

dbutils.notebook.exit(json.dumps({"resumen": resumen.to_dict("records"),
                                  "aciertos_segun_humano": int(hoja["humano"].sum()),
                                  "desacuerdos": desacuerdos.to_dict("records")}, ensure_ascii=False))
