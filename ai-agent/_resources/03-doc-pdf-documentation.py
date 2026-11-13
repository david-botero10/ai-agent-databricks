# Databricks notebook source
# MAGIC %md
# MAGIC # Generación de la documentación PDF (base de conocimiento)
# MAGIC
# MAGIC Genera manuales técnicos en PDF de equipos y servicios de agua, energía y gas usando un LLM del propio workspace
# MAGIC y los guarda en el volumen `pdf_documentation`. Luego, el notebook `03.1-pdf-rag-tool` los parsea
# MAGIC y los indexa en Vector Search.
# MAGIC
# MAGIC **Cambios para Databricks Free Edition / serverless:**
# MAGIC - Se elimina la URL fija de otro workspace: el LLM se llama con el cliente de despliegue de MLflow,
# MAGIC   que usa automáticamente el host y las credenciales del workspace actual.
# MAGIC - Se reemplaza WeasyPrint (requiere librerías de sistema que no están en serverless) por `fpdf2`, 100% Python.
# MAGIC - Se generan pocos documentos (widget `num_docs`) para limitar el consumo de la cuota gratuita.
# MAGIC - Los manuales se generan en **inglés** porque el modelo de embeddings (`gte-large-en`) es para inglés.
# MAGIC   El agente responde en español igualmente.

# COMMAND ----------

# MAGIC %pip install fpdf2 "mlflow>=3.10.1"
# MAGIC %restart_python

# COMMAND ----------

# MAGIC %run ../config

# COMMAND ----------

# Número de manuales a generar (los primeros de la lista de prioridad incluyen los usados en los ejemplos)
dbutils.widgets.text("num_docs", "8", "Número de manuales a generar")
NUM_DOCS = int(dbutils.widgets.get("num_docs"))

# COMMAND ----------

import os
import re
import shutil
import tempfile
import unicodedata
from concurrent.futures import ThreadPoolExecutor

def get_documentation_entities():
    """Lista de manuales a generar (equipos y servicios de agua, energía y gas). Se mantienen en inglés porque
    el modelo de embeddings del índice (gte-large-en) está optimizado para inglés."""

    return [
        # Equipos de medición (hardware)
        {
            "title": "Smart Electricity Meter User Manual",
            "category": "Hardware",
            "model": "MED-ENERGIA-2024",
            "description": "Residential smart electricity meter: display, readings, reset and pairing procedures"
        },
        {
            "title": "Water Meter Installation and Reading Guide",
            "category": "Hardware",
            "model": "MED-AGUA-2024",
            "description": "Water meter installation, reading, leak detection and maintenance"
        },
        {
            "title": "Prepaid Meter User Manual",
            "category": "Hardware",
            "model": "PREPAGO-2024",
            "description": "Prepaid energy meter: recharge, balance display, low-balance alerts and disconnection behavior"
        },

        # Documentación técnica
        {
            "title": "Meter Error Codes Reference",
            "category": "Technical",
            "model": "ERR-MED-2024",
            "description": "Complete reference of meter display error codes, symptoms, causes and resolution steps"
        },
        {
            "title": "Service Connection and Metering Regulation",
            "category": "Technical",
            "model": "ACOMETIDA-2024",
            "description": "New service connections, technical requirements, inspection and metering rules"
        },

        # Tarifas y facturación
        {
            "title": "Residential Tariffs and Subsidies Guide",
            "category": "Billing",
            "model": "TARIFA-RES-2024",
            "description": "Residential tariff structure, subsidies, consumption brackets and billing cycle"
        },
        {
            "title": "Commercial and Industrial Tariffs Guide",
            "category": "Billing",
            "model": "TARIFA-COM-2024",
            "description": "Commercial and industrial tariff schemes, demand charges and contract conditions"
        },

        # Políticas de servicio
        {
            "title": "Reconnection and Suspension Policy",
            "category": "Policy",
            "model": "RECONEXION-2024",
            "description": "Suspension for non-payment, reconnection steps, fees and payment agreements"
        },
        {
            "title": "Emergency and Outage Response Protocol",
            "category": "Operations",
            "model": "EMERG-2024",
            "description": "Outage reporting, emergency response times, restoration priorities and compensation"
        },
        {
            "title": "Customer Requests, Complaints and Claims Procedure",
            "category": "Customer Service",
            "model": "PQR-2024",
            "description": "Customer request, complaint and claim (PQR) procedure, response deadlines and escalation levels"
        },
        {
            "title": "Energy Saving and Efficient Use Guide",
            "category": "Customer Service",
            "model": "AHORRO-2024",
            "description": "Practical energy saving recommendations, consumption benchmarks and efficiency programs"
        }
    ]



# Manuales que se generan primero porque se usan en los ejemplos de los notebooks
PRIORITY_MODELS = ["MED-ENERGIA-2024", "ERR-MED-2024", "PQR-2024", "TARIFA-RES-2024",
                   "RECONEXION-2024", "PREPAGO-2024", "ACOMETIDA-2024", "EMERG-2024"]


def get_prioritized_entities(n):
    """Devuelve los n primeros manuales, poniendo primero los de PRIORITY_MODELS."""
    entities = get_documentation_entities()
    entities.sort(key=lambda e: PRIORITY_MODELS.index(e["model"]) if e["model"] in PRIORITY_MODELS else 999)
    return entities[:n]

# COMMAND ----------

SYSTEM_PROMPT = """You are an expert technical documentation writer for water, energy and gas utility equipment and services.
Write in English, with a professional tone. Output plain text using this light markdown only:
'# ' for the document title, '## ' for sections, '### ' for sub-sections, '- ' for bullets and '1. ' for numbered steps.
Do not use tables, code fences, bold markers or HTML."""


def get_documentation_prompt(entity):
    """Construye el prompt para un manual concreto."""
    return f"""Write a complete documentation manual (about 1500-2500 words) for:

TITLE: {entity['title']}
CATEGORY: {entity['category']}
MODEL: {entity['model']}
DESCRIPTION: {entity['description']}

The company is "ServiCiudad", a public utility that provides ELECTRICITY, WATER and GAS to
residential and business customers. It is NOT a telecommunications company: never mention
telecom, internet service providers, routers, fiber or mobile plans.

The manual will be used as the only source of a RAG customer-support assistant, so be explicit and self-contained.
Mention the model name ({entity['model']}) in every section. Include, when applicable:
- Overview and technical specifications (with concrete numbers)
- Installation and setup (numbered steps)
- How to restart / reset the device or service, step by step
- Error code reference: at least 6 error codes (for example Error 01, Error 1001...), each with description, symptoms, cause, numbered resolution steps and escalation policy
- Troubleshooting scenarios
- Firmware / maintenance procedures
- Warranty, return and refund policy
- FAQ with at least 8 questions and answers
- Support and escalation contacts"""


def call_llm_api(prompt: str) -> str:
    """Llama al LLM configurado (LLM_ENDPOINT_NAME) usando las credenciales del workspace actual."""
    from mlflow.deployments import get_deploy_client
    client = get_deploy_client("databricks")
    response = client.predict(
        endpoint=LLM_ENDPOINT_NAME,
        inputs={
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.2,
            "max_tokens": 4000,
        },
    )
    content = response["choices"][0]["message"]["content"]
    # Algunos modelos devuelven el contenido como lista de bloques (p. ej. razonamiento + texto)
    if isinstance(content, list):
        content = "\n".join(c.get("text", "") for c in content if isinstance(c, dict) and c.get("type") == "text")
    return content

# COMMAND ----------

def to_latin1(text: str) -> str:
    """Las fuentes base de fpdf2 solo soportan latin-1: reemplazamos los caracteres no soportados."""
    replacements = {"–": "-", "—": "-", "‘": "'", "’": "'", "“": '"', "”": '"',
                    "•": "-", "…": "...", " ": " ", "→": "->", "≤": "<=", "≥": ">="}
    for k, v in replacements.items():
        text = text.replace(k, v)
    text = unicodedata.normalize("NFKD", text)
    return text.encode("latin-1", "ignore").decode("latin-1")


def break_long_words(line: str, max_len: int = 60) -> str:
    """Corta palabras muy largas (URLs, comandos) para que fpdf2 pueda ajustarlas al ancho de página."""
    return " ".join(w if len(w) <= max_len else " ".join(w[i:i + max_len] for i in range(0, len(w), max_len))
                    for w in line.split(" "))


def convert_text_to_pdf(text: str, output_path: str):
    """Convierte el texto (markdown ligero) a PDF con fpdf2 y lo guarda en output_path."""
    from fpdf import FPDF
    pdf = FPDF(format="A4")
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    for raw in to_latin1(text).splitlines():
        line = break_long_words(raw.strip().replace("**", "").replace("`", ""))
        if not line:
            pdf.ln(2)  # línea en blanco = pequeño espacio vertical
            continue
        # Estilo según el tipo de línea: título, sección, sub-sección o texto normal
        if line.startswith("# "):
            pdf.set_font("Helvetica", "B", 18); line = line[2:]
        elif line.startswith("## "):
            pdf.ln(3); pdf.set_font("Helvetica", "B", 14); line = line[3:]
        elif line.startswith("### "):
            pdf.set_font("Helvetica", "B", 12); line = line[4:]
        else:
            pdf.set_font("Helvetica", "", 10)
        pdf.multi_cell(0, 5, line, new_x="LMARGIN", new_y="NEXT")
    # Escribimos primero en disco local y luego copiamos al volumen de UC
    tmp = os.path.join(tempfile.mkdtemp(), os.path.basename(output_path))
    pdf.output(tmp)
    shutil.copy(tmp, output_path)
    print(f"  ✓ PDF guardado: {output_path}")

# COMMAND ----------

def generate_one(entity, folder):
    """Genera un manual: llama al LLM y convierte la respuesta a PDF."""
    try:
        text = call_llm_api(get_documentation_prompt(entity))
        filename = f"{entity['model'].replace('-', '_').lower()}_manual.pdf"  # p. ej. adsl_r500_manual.pdf
        convert_text_to_pdf(text, f"{folder}/{filename}")
    except Exception as e:
        print(f"  ✗ Error generando {entity['title']}: {str(e)[:300]}")


def generate_documentation_manuals(path: str, n: int):
    """Genera n manuales en paralelo (4 a la vez) dentro de {path}/pdf_documentation."""
    folder = f"{path}/pdf_documentation"
    dbutils.fs.mkdirs(folder)
    entities = get_prioritized_entities(n)
    print(f"Generando {len(entities)} manuales con el modelo {LLM_ENDPOINT_NAME}...")
    with ThreadPoolExecutor(max_workers=4) as executor:
        list(executor.map(lambda e: generate_one(e, folder), entities))
    print(f"\n¡Documentación generada! Archivos guardados en '{folder}'.")

# COMMAND ----------

path = f'/Volumes/{catalog}/{schema}/{volume_name}'
generate_documentation_manuals(path, NUM_DOCS)

# Mostramos los archivos generados
display(dbutils.fs.ls(f"{path}/pdf_documentation"))
