# Databricks notebook source
# MAGIC %md
# MAGIC # Parte 6 — Promover la mejor configuración a producción
# MAGIC
# MAGIC Registra en Unity Catalog el agente con la configuración ganadora de los experimentos (`03.2-experimentos`)
# MAGIC y le asigna el alias `model-to-deploy`, que es el que usan el endpoint, la demo y el quality gate.
# MAGIC
# MAGIC | Configuración | Correctness | Multi-documento | Tokens reales por pregunta | Latencia p50 |
# MAGIC |---|---|---|---|---|
# MAGIC | base (k = 3) | 0.53 | 0/3 | 7.536 | 4,9 s |
# MAGIC | **k = 6 (promovida)** | **0.67** | **1/3** | 11.758 | 5,5 s |
# MAGIC
# MAGIC k = 6 es la única configuración que responde una pregunta multi-documento, el fallo dominante (recuperación de
# MAGIC un segundo manual), con latencia similar y +56 % de tokens.

# COMMAND ----------

# MAGIC %pip install "mlflow>=3.10.1" databricks-langchain databricks-agents "unitycatalog-langchain[databricks]" "typing_extensions>=4.13" "langgraph-prebuilt==1.0.8" databricks-vectorsearch

# COMMAND ----------

# Reinicia Python para cargar las versiones recién instaladas (p. ej. typing_extensions)
dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %run ./_resources/01-setup

# COMMAND ----------

import importlib.metadata
import os
import sys

import mlflow
import yaml
from mlflow import MlflowClient

dbutils.widgets.text("retriever_k", "6")
RETRIEVER_K = int(dbutils.widgets.get("retriever_k"))

AGENT_DIR = os.path.join(os.getcwd(), "02-agent-eval")
sys.path.append(AGENT_DIR)
CONF_PATH = os.path.join(AGENT_DIR, "agent_config.yaml")
mlflow.set_experiment(AGENT_DIR + "/02.1_agent_evaluation")

with open(os.path.join(os.getcwd(), "prompts", "system_prompt.txt"), encoding="utf-8") as f:
    SYSTEM_PROMPT = f.read().strip()

config = {
    "config_version_name": f"production_k{RETRIEVER_K}",
    "input_example": [{"role": "user", "content": "¿Qué significa el código de error 4001 de mi medidor ERR-MED-2024?"}],
    "uc_tool_names": [f"{catalog}.{dbName}.*"],
    "system_prompt": SYSTEM_PROMPT,
    "llm_endpoint_name": LLM_ENDPOINT_NAME,
    "max_history_messages": 20,
    "retriever_config": {
        "index_name": f"{catalog}.{dbName}.knowledge_base_vs_index",
        "tool_name": "product_technical_docs_retriever",
        "num_results": RETRIEVER_K,
        "description": "Recupera documentación interna sobre nuestros servicios públicos (energía, agua y gas): medidores, tarifas, facturación, reconexiones, emergencias y trámites. Usa esta herramienta para cualquier pregunta sobre documentación, procedimientos o problemas con el servicio. IMPORTANTE: escribe siempre la consulta (query) en inglés, porque la documentación está en inglés.",
    },
    "mcp_server_urls": [],
}
with open(CONF_PATH, "w", encoding="utf-8") as f:
    yaml.dump(config, f, allow_unicode=True)

# agent.py reads "../02-agent-eval/agent_config.yaml" relative to the working directory: run it from its folder.
os.chdir(AGENT_DIR)
from agent import AGENT  # noqa: E402

# COMMAND ----------

with mlflow.start_run(run_name=f"production_k{RETRIEVER_K}") as run:
    logged = mlflow.pyfunc.log_model(
        name="agent",
        python_model=os.path.join(AGENT_DIR, "agent.py"),
        model_config=CONF_PATH,
        input_example={"input": config["input_example"]},
        resources=AGENT.get_resources(),
        extra_pip_requirements=["databricks-connect", "langgraph-prebuilt==1.0.8", "databricks-vectorsearch",
                                f"mcp=={importlib.metadata.version('mcp')}"],
    )
    mlflow.log_params({"retriever_k": RETRIEVER_K, "index": "knowledge_base_vs_index (documento completo)"})

UC_MODEL_NAME = f"{catalog}.{dbName}.{MODEL_NAME}"
version = mlflow.register_model(model_uri=logged.model_uri, name=UC_MODEL_NAME,
                                tags={"retriever_k": str(RETRIEVER_K), "promoted_by": "03.2-experimentos"})
client = MlflowClient()
client.set_registered_model_alias(UC_MODEL_NAME, "model-to-deploy", version.version)
client.set_model_version_tag(UC_MODEL_NAME, version.version, "justificacion",
                             "Unica configuracion que responde una pregunta multi-documento; correctness 0.67 vs 0.53; "
                             "latencia similar; +56% tokens")

# Prueba rápida del modelo promovido
loaded = mlflow.pyfunc.load_model(f"models:/{UC_MODEL_NAME}@model-to-deploy")
import pandas as pd  # noqa: E402
respuesta = loaded.predict(pd.DataFrame({"input": [[{"role": "user", "content": "¿Qué significa el código de error 4001 de mi medidor ERR-MED-2024?"}]]}))
texto = respuesta["output"][-1]["content"][-1]["text"]
print(f"Versión {version.version} promovida con alias model-to-deploy (k={RETRIEVER_K})\n{texto}")
dbutils.notebook.exit(f"version={version.version} k={RETRIEVER_K} | {texto[:300]}")
