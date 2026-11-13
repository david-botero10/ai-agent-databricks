# Databricks notebook source
# MAGIC %md
# MAGIC # Experimentos de la Parte 3: prompt, k y chunking
# MAGIC
# MAGIC Cada variante cambia **una sola variable** respecto a la configuración base:
# MAGIC
# MAGIC | Variante | Prompt | k | Índice |
# MAGIC |---|---|---|---|
# MAGIC | `base` | v2 (producción) | 3 | documento completo |
# MAGIC | `prompt_v1` | v1 (genérico) | 3 | documento completo |
# MAGIC | `k2` / `k6` | v2 | 2 / 6 | documento completo |
# MAGIC | `chunk256` / `chunk1024` | v2 | 3 | fragmentos de 256 / 1024 caracteres |
# MAGIC
# MAGIC Todas usan el dataset curado de 15 preguntas y los mismos jueces. Los resultados por pregunta se guardan
# MAGIC en el volumen (`experimentos/`) para diagnosticar los fallos con su traza.

# COMMAND ----------

# MAGIC %pip install "mlflow>=3.10.1" databricks-langchain databricks-agents "unitycatalog-langchain[databricks]" "typing_extensions>=4.13" "langgraph-prebuilt==1.0.8"

# COMMAND ----------

# Reinicia Python para cargar las versiones recién instaladas (p. ej. typing_extensions)
dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %run ../_resources/01-setup

# COMMAND ----------

dbutils.widgets.text("variants", "base,prompt_v1,k2,k6,chunk256,chunk1024")
VARIANTS = [v.strip() for v in dbutils.widgets.get("variants").split(",") if v.strip()]
print("Variantes a evaluar:", VARIANTS)

# COMMAND ----------

import json
import os
import sys

import mlflow
import pandas as pd
from databricks.vector_search.client import VectorSearchClient

agent_eval_path = os.path.abspath(os.path.join(os.getcwd(), "../02-agent-eval"))
sys.path.append(agent_eval_path)
mlflow.set_experiment(agent_eval_path + "/02.1_agent_evaluation")

from agent import LangGraphResponsesAgent  # noqa: E402

vsc = VectorSearchClient(disable_notice=True)
OUT_DIR = f"/Volumes/{catalog}/{dbName}/{volume_name}/experimentos"
os.makedirs(OUT_DIR, exist_ok=True)

PROMPT_V1 = "Tu trabajo es ayudar a los clientes. Llama a la herramienta para responder."
with open(os.path.abspath(os.path.join(os.getcwd(), "../prompts/system_prompt.txt")), encoding="utf-8") as f:
    PROMPT_V2 = f.read().strip()

FULL_DOC_INDEX = f"{catalog}.{dbName}.knowledge_base_vs_index"
RETRIEVER_DESCRIPTION = (
    "Recupera documentación interna sobre nuestros servicios públicos (energía, agua y gas): medidores, tarifas, "
    "facturación, reconexiones, emergencias y trámites. Usa esta herramienta para cualquier pregunta sobre "
    "documentación, procedimientos o problemas con el servicio. IMPORTANTE: escribe siempre la consulta (query) "
    "en inglés, porque la documentación está en inglés."
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Índices con chunking (solo si se piden variantes `chunk*`)

# COMMAND ----------

def split_text(text, size, overlap):
    """Divide el texto en fragmentos de ~`size` caracteres sin cortar palabras, con solapamiento."""
    words, chunks, current = text.split(), [], ""
    for word in words:
        if current and len(current) + 1 + len(word) > size:
            chunks.append(current)
            current = current[-overlap:].split(" ", 1)[-1] if overlap else ""
        current = f"{current} {word}".strip()
    if current:
        chunks.append(current)
    return chunks


def build_chunk_index(size):
    """Crea la tabla knowledge_base_chunks_<size> y su índice de Vector Search (si no existen)."""
    table = f"{catalog}.{dbName}.knowledge_base_chunks_{size}"
    index = f"{table}_vs_index"
    kb = spark.table(f"{catalog}.{dbName}.knowledge_base").select("product_name", "title", "content", "doc_uri").toPandas()
    rows = []
    for _, doc in kb.iterrows():
        for i, chunk in enumerate(split_text(doc["content"], size, overlap=size // 10)):
            # El nombre del producto va en cada fragmento para que la búsqueda siga encontrando el documento correcto
            rows.append({"product_name": doc["product_name"], "title": doc["title"],
                         "content": f"[{doc['product_name']}] {chunk}", "doc_uri": doc["doc_uri"], "chunk_id": i})
    spark.sql(f"""CREATE TABLE IF NOT EXISTS {table} (
        id BIGINT GENERATED ALWAYS AS IDENTITY, product_name STRING, title STRING,
        content STRING, doc_uri STRING, chunk_id INT) TBLPROPERTIES (delta.enableChangeDataFeed = true)""")
    spark.createDataFrame(pd.DataFrame(rows)).createOrReplaceTempView("chunks_tmp")
    spark.sql(f"INSERT OVERWRITE TABLE {table} (product_name, title, content, doc_uri, chunk_id) SELECT * FROM chunks_tmp")
    print(f"{table}: {len(rows)} fragmentos a partir de {len(kb)} documentos")
    if not index_exists(vsc, VECTOR_SEARCH_ENDPOINT_NAME, index):
        vsc.create_delta_sync_index(endpoint_name=VECTOR_SEARCH_ENDPOINT_NAME, index_name=index,
                                    source_table_name=table, pipeline_type="TRIGGERED", primary_key="id",
                                    embedding_source_column="content",
                                    embedding_model_endpoint_name=EMBEDDING_ENDPOINT_NAME)
    else:
        wait_for_index_to_be_ready(vsc, VECTOR_SEARCH_ENDPOINT_NAME, index)
        vsc.get_index(VECTOR_SEARCH_ENDPOINT_NAME, index).sync()
    wait_for_index_to_be_ready(vsc, VECTOR_SEARCH_ENDPOINT_NAME, index)
    return index


CHUNK_INDEXES = {}
for v in VARIANTS:
    if v.startswith("chunk"):
        size = int(v.replace("chunk", ""))
        CHUNK_INDEXES[v] = build_chunk_index(size)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Evaluación de cada variante

# COMMAND ----------

BASE = {"prompt": PROMPT_V2, "k": 3, "index": FULL_DOC_INDEX}
SPECS = {
    "base": {},
    "prompt_v1": {"prompt": PROMPT_V1},
    "k2": {"k": 2},
    "k6": {"k": 6},
    "chunk256": {"index": CHUNK_INDEXES.get("chunk256")},
    "chunk1024": {"index": CHUNK_INDEXES.get("chunk1024")},
}

eval_dataset = mlflow.genai.datasets.get_dataset(f"{catalog}.{dbName}.ai_agent_mlflow_eval")
scorers = get_scorers(JUDGE_MODEL)


def trace_costs(run_id):
    """Latencia (ms) y tokens totales por pregunta, leídos de las trazas del run."""
    traces = mlflow.search_traces(run_id=run_id)
    lat = traces["execution_duration"].median() if "execution_duration" in traces else None
    tokens = None
    try:
        usage = [json.loads(m.get("mlflow.trace.tokenUsage", "{}")).get("total_tokens")
                 for m in traces["trace_metadata"]]
        usage = [u for u in usage if u]
        tokens = sum(usage) / len(usage) if usage else None
    except Exception as e:
        print("sin tokens en las trazas:", str(e)[:100])
    return lat, tokens


summary = []
for name in VARIANTS:
    cfg = {**BASE, **SPECS[name]}
    agent = LangGraphResponsesAgent(
        uc_tool_names=[f"{catalog}.{dbName}.*"],
        llm_endpoint_name=LLM_ENDPOINT_NAME,
        system_prompt=cfg["prompt"],
        retriever_config={"index_name": cfg["index"], "tool_name": "product_technical_docs_retriever",
                          "num_results": cfg["k"], "description": RETRIEVER_DESCRIPTION},
        max_history_messages=20,
    )

    def make_predict_fn(_agent):
        def predict_fn(question):
            response = _agent.predict({"input": [{"role": "user", "content": question}]}).model_dump()
            return response["output"][-1]["content"][-1]["text"]
        return predict_fn

    predict_fn = make_predict_fn(agent)

    with mlflow.start_run(run_name=f"exp_{name}") as run:
        mlflow.log_params({"variant": name, "retriever_k": cfg["k"], "index": cfg["index"],
                           "prompt_version": "v1" if cfg["prompt"] == PROMPT_V1 else "v2"})
        results = mlflow.genai.evaluate(data=eval_dataset, predict_fn=predict_fn, scorers=scorers)
    lat, tokens = trace_costs(run.info.run_id)
    row = {"variant": name, "run_id": run.info.run_id, "k": cfg["k"], "index": cfg["index"].split(".")[-1],
           "latency_ms_p50": lat, "tokens_per_question": tokens,
           **{k: round(v, 3) for k, v in results.metrics.items() if k.endswith("/mean")}}
    summary.append(row)
    print(row)
    # Resultados por pregunta para el diagnóstico de fallos
    df = results.result_df.copy()
    df.astype(str).to_csv(f"{OUT_DIR}/{name}.csv", index=False)

summary_df = pd.DataFrame(summary)
summary_df.to_csv(f"{OUT_DIR}/resumen.csv", index=False)
display(summary_df)

# COMMAND ----------

dbutils.notebook.exit(summary_df.to_json(orient="records"))
