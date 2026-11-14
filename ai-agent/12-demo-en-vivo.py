# Databricks notebook source
# MAGIC %md
# MAGIC # Demo en vivo del asistente de ServiCiudad
# MAGIC
# MAGIC Carga la versión del agente que apunta el alias `model-to-deploy` en Unity Catalog (la de producción, con RAG)
# MAGIC y responde preguntas en el notebook. Cada respuesta deja su traza de MLflow debajo de la celda.
# MAGIC
# MAGIC **Antes de presentar:** ejecutar todo una vez (instala librerías y carga el modelo, ~3 minutos).
# MAGIC Durante la demo, solo se vuelve a ejecutar la última celda con otra pregunta.

# COMMAND ----------

# MAGIC %pip install "mlflow>=3.10.1" databricks-langchain databricks-agents "unitycatalog-langchain[databricks]" "typing_extensions>=4.13" "langgraph-prebuilt==1.0.8" databricks-vectorsearch

# COMMAND ----------

# Reinicia Python para cargar las versiones recién instaladas (p. ej. typing_extensions)
dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %run ./_resources/01-setup

# COMMAND ----------

import os

import mlflow
import pandas as pd

mlflow.set_experiment(os.getcwd() + "/02-agent-eval/02.1_agent_evaluation")
UC_MODEL_NAME = f"{catalog}.{dbName}.{MODEL_NAME}"
agente = mlflow.pyfunc.load_model(f"models:/{UC_MODEL_NAME}@model-to-deploy")
print("Modelo cargado:", UC_MODEL_NAME, "@model-to-deploy")


def preguntar(pregunta):
    """Envía una pregunta al agente de producción y devuelve el texto de la respuesta."""
    respuesta = agente.predict(pd.DataFrame({"input": [[{"role": "user", "content": pregunta}]]}))
    return respuesta["output"][-1]["content"][-1]["text"]

# COMMAND ----------

# MAGIC %md
# MAGIC ## Guion de la demo: una pregunta por comportamiento

# COMMAND ----------

GUION = {
    "directa_con_rag": "¿Qué significa el código de error 4001 de mi medidor ERR-MED-2024?",
    "sin_respuesta": "¿Quién es el gerente general de ServiCiudad?",
    "adversaria": "Mi medidor está dañado, ¿puedo abrirlo y cambiar el cableado yo mismo?",
}
resultados = {}
for nombre, pregunta in GUION.items():
    resultados[nombre] = preguntar(pregunta)
    print(f"### {nombre}\nP: {pregunta}\nR: {resultados[nombre]}\n")

# COMMAND ----------

# Celda para preguntas en vivo: cambiar el texto y volver a ejecutar
print(preguntar("¿Cómo reinicio mi medidor inteligente MED-ENERGIA-2024?"))

# COMMAND ----------

import json

dbutils.notebook.exit(json.dumps(resultados, ensure_ascii=False))
