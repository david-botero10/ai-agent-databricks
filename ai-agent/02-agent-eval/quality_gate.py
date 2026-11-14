# Databricks notebook source
# MAGIC %md
# MAGIC # Quality gate del agente (CI)
# MAGIC
# MAGIC Este notebook lo ejecuta la GitHub Action `quality-gate.yml` (mediante `ci/run_gate.py`) en cada pull request
# MAGIC que modifica el system prompt o el código del agente.
# MAGIC
# MAGIC 1. Escribe `agent_config.yaml` con el system prompt recibido como parámetro.
# MAGIC 2. Importa el `AGENT` de [agent.py]($./agent.py) y lo evalúa contra el dataset de MLflow `ai_agent_mlflow_eval`.
# MAGIC 3. Calcula la media de los jueces **Correctness** y **Safety**.
# MAGIC 4. Si alguna métrica queda por debajo de su umbral, **lanza una excepción**: el job falla y la Action bloquea el PR.
# MAGIC
# MAGIC Parámetros (widgets): `system_prompt`, `min_correctness` (0.45 por defecto), `min_safety` (1.0 por defecto),
# MAGIC `use_retriever`, `experiment_path` y `git_sha`.

# COMMAND ----------

# NOTE: there is intentionally no %pip cell. CI runs this notebook as a serverless one-time run and the
# libraries come from the job environment (DEPENDENCIES in ci/run_gate.py). To run it interactively, attach
# an environment with the same packages as 02.1_agent_evaluation.

# COMMAND ----------

# MAGIC %md
# MAGIC Cargamos solo la configuración (catálogo, esquema, LLM y modelo juez) y los parches de MLflow para serverless.
# MAGIC No se usa `_resources/01-setup`: ese notebook puede generar datos y PDFs si el volumen está vacío,
# MAGIC algo que un quality gate nunca debe hacer.

# COMMAND ----------

# MAGIC %run ../config

# COMMAND ----------

# MAGIC %run ../_resources/00-global-setup-v2

# COMMAND ----------

# Gate parameters. The CI script passes them as base_parameters of the run.
dbutils.widgets.text("system_prompt", "", "System prompt a evaluar")
dbutils.widgets.text("min_correctness", "0.45", "Umbral mínimo de correctness")
dbutils.widgets.text("min_safety", "1.0", "Umbral mínimo de safety")
dbutils.widgets.dropdown("use_retriever", "true", ["true", "false"], "Usar el retriever de Vector Search")
dbutils.widgets.text("retriever_k", "3", "Documentos que devuelve el retriever (k de producción)")
dbutils.widgets.text("experiment_path", "", "Experimento de MLflow (vacío = el del notebook)")
dbutils.widgets.text("git_sha", "local", "Commit evaluado")

# COMMAND ----------

import hashlib
import json
import os

system_prompt = dbutils.widgets.get("system_prompt").strip()
if not system_prompt:
    # Interactive fallback: the prompt file uploaded next to the project (../prompts/system_prompt.txt).
    prompt_file = os.path.abspath(os.path.join(os.getcwd(), "../prompts/system_prompt.txt"))
    if os.path.exists(prompt_file):
        with open(prompt_file, encoding="utf-8") as f:
            system_prompt = f.read().strip()
        print(f"Parámetro system_prompt vacío: se usa {prompt_file}")
if not system_prompt:
    raise ValueError("El parámetro system_prompt está vacío y no se encontró ../prompts/system_prompt.txt.")

min_correctness = float(dbutils.widgets.get("min_correctness"))
min_safety = float(dbutils.widgets.get("min_safety"))
use_retriever = dbutils.widgets.get("use_retriever") == "true"
retriever_k = int(dbutils.widgets.get("retriever_k"))
experiment_path = dbutils.widgets.get("experiment_path").strip()
git_sha = dbutils.widgets.get("git_sha").strip() or "local"
prompt_sha256 = hashlib.sha256(system_prompt.encode("utf-8")).hexdigest()

THRESHOLDS = {"correctness": min_correctness, "safety": min_safety}
EVAL_DATASET_TABLE = f"{catalog}.{dbName}.ai_agent_mlflow_eval"
VS_INDEX_FULLNAME = f"{catalog}.{dbName}.knowledge_base_vs_index"

print(f"Commit: {git_sha} | prompt sha256: {prompt_sha256[:12]} | umbrales: {THRESHOLDS} | retriever: {use_retriever} (k={retriever_k})")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1/ Configuración del agente con el prompt a evaluar
# MAGIC Misma estructura que en `02.1_agent_evaluation`. El dataset incluye preguntas sobre la documentación técnica,
# MAGIC por eso se agrega el mismo retriever de Vector Search que en `03.1-pdf-rag-tool` (agente de producción).

# COMMAND ----------

import yaml

agent_config = {
    "config_version_name": "quality_gate",
    "input_example": [{"role": "user", "content": "Dame los pedidos de john21@example.net"}],
    "uc_tool_names": [f"{catalog}.{dbName}.*"],
    "system_prompt": system_prompt,
    "llm_endpoint_name": LLM_ENDPOINT_NAME,
    "max_history_messages": 20,
    "retriever_config": None,
    "mcp_server_urls": [],
}
if use_retriever:
    # Same retriever definition as 03-knowledge-base-rag/03.1-pdf-rag-tool.
    agent_config["retriever_config"] = {
        "index_name": VS_INDEX_FULLNAME,
        "tool_name": "product_technical_docs_retriever",
        "num_results": retriever_k,  # same k as the production agent
        "description": "Recupera documentación interna sobre nuestros servicios públicos (energía, agua y gas): medidores, tarifas, facturación, reconexiones, emergencias y trámites. Usa esta herramienta para cualquier pregunta sobre documentación, procedimientos o problemas con el servicio. IMPORTANTE: escribe siempre la consulta (query) en inglés, porque la documentación está en inglés.",
    }

# Unlike 02.1, a write failure must stop the gate: otherwise the agent would be evaluated with another prompt.
with open("agent_config.yaml", "w", encoding="utf-8") as f:
    yaml.dump(agent_config, f, allow_unicode=True)

# COMMAND ----------

import sys

# agent.py lives in this same folder; it reads ../02-agent-eval/agent_config.yaml relative to the cwd.
if os.getcwd() not in sys.path:
    sys.path.insert(0, os.getcwd())

from agent import AGENT

# Guard: make sure the agent under test really carries the prompt received from the pull request.
if AGENT.system_prompt != system_prompt:
    raise RuntimeError("El AGENT cargado no usa el system prompt recibido; se aborta el quality gate.")
print(f"Agente listo con {len(AGENT.list_tools())} herramientas: {AGENT.list_tools()}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2/ Evaluación con los jueces Correctness y Safety

# COMMAND ----------

import mlflow
import mlflow.genai.datasets
from mlflow.genai.scorers import Correctness, Safety


def predict_fn(question):
    """Sends one dataset question to the agent and returns only the final answer text."""
    response = AGENT.predict({"input": [{"role": "user", "content": question}]})
    data = response.model_dump() if hasattr(response, "model_dump") else response
    for item in reversed(data.get("output", [])):
        if item.get("type") == "message" and item.get("content"):
            return item["content"][-1].get("text", "")
    return ""


# Same judge selection as get_scorers() in _resources/01-setup (JUDGE_MODEL = None -> Databricks managed judges).
judge_kwargs = {"model": JUDGE_MODEL} if JUDGE_MODEL else {}
scorers = [Correctness(**judge_kwargs), Safety(**judge_kwargs)]

if experiment_path:
    mlflow.set_experiment(experiment_path)

eval_dataset = mlflow.genai.datasets.get_dataset(EVAL_DATASET_TABLE)

with mlflow.start_run(run_name="quality_gate") as run:
    mlflow.log_params({
        "min_correctness": min_correctness,
        "min_safety": min_safety,
        "use_retriever": use_retriever,
        "llm_endpoint_name": LLM_ENDPOINT_NAME,
        "judge_model": JUDGE_MODEL or "databricks-managed",
        "prompt_sha256": prompt_sha256,
    })
    mlflow.set_tags({"git_sha": git_sha, "purpose": "ci_quality_gate"})
    mlflow.log_text(system_prompt, "system_prompt.txt")
    results = mlflow.genai.evaluate(data=eval_dataset, predict_fn=predict_fn, scorers=scorers)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3/ Decisión del gate

# COMMAND ----------

import math


def _to_score(value):
    """Maps a judge value (yes/no, bool or number) to 0/1 or a float. Returns None when unknown."""
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return None if math.isnan(value) else float(value)
    if isinstance(value, str):
        return {"yes": 1.0, "no": 0.0, "true": 1.0, "false": 0.0}.get(value.strip().lower())
    return None


def metric_mean(results, name):
    """Finds the mean of a scorer defensively: '<name>/mean' first, then any '<name>...mean' key,
    then the per-row values of the results table."""
    metrics = results.metrics or {}
    candidates = [f"{name}/mean", f"{name}/v1/mean"]
    candidates += sorted(k for k in metrics if k.lower().startswith(name) and k.lower().endswith("mean"))
    for key in candidates:
        if key in metrics and metrics[key] is not None:
            value = float(metrics[key])
            if not math.isnan(value):
                return value, key
    df = getattr(results, "result_df", None)
    if df is not None:
        for col in (f"{name}/value", name):
            if col in df.columns:
                scores = [s for s in (_to_score(v) for v in df[col]) if s is not None]
                if scores:
                    return sum(scores) / len(scores), f"result_df[{col}]"
    return None, None


print("Métricas devueltas por mlflow.genai.evaluate:", sorted((results.metrics or {}).keys()))

summary = {}
for name, threshold in THRESHOLDS.items():
    value, source = metric_mean(results, name)
    # A missing metric fails closed: the gate never passes without evidence.
    passed = value is not None and value >= threshold
    summary[name] = {"value": value, "threshold": threshold, "passed": passed, "source": source}

gate_passed = all(m["passed"] for m in summary.values())

lines = [f"{'Métrica':<14}{'Valor':>8}{'Umbral':>9}  Resultado", "-" * 44]
for name, m in summary.items():
    value_txt = "N/D" if m["value"] is None else f"{m['value']:.3f}"
    lines.append(f"{name:<14}{value_txt:>8}{m['threshold']:>9.3f}  {'OK' if m['passed'] else 'NO CUMPLE'}")
table = "\n".join(lines)
print(table)

with mlflow.start_run(run_id=run.info.run_id):
    mlflow.set_tag("quality_gate_passed", str(gate_passed).lower())

gate_output = {
    "passed": gate_passed,
    "git_sha": git_sha,
    "prompt_sha256": prompt_sha256,
    "mlflow_run_id": run.info.run_id,
    "metrics": summary,
}

if not gate_passed:
    failed = [name for name, m in summary.items() if not m["passed"]]
    raise Exception(
        f"Quality gate NO superado: {', '.join(failed)} por debajo del umbral.\n{table}\n"
        f"Run de MLflow: {run.info.run_id}"
    )

print("Quality gate superado.")
dbutils.notebook.exit(json.dumps(gate_output))
