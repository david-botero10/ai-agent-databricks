# Asistente de Servicios Públicos con Databricks y MLflow 3

Implementación del **desafío de GenAIOps** sobre el stack de Databricks: un agente RAG para una
empresa ficticia de servicios públicos (**ServiCiudad**: agua, energía y gas), construido con
LangGraph, registrado en MLflow 3 y evaluado con jueces LLM.

> Este repositorio es un **fork** de [`darkanita/ai-agent-databricks`](https://github.com/darkanita/ai-agent-databricks),
> la adaptación a Databricks Free Edition de la demo `ai-agent` de dbdemos.
>
> El desafío original (`DESAFIO.md`) está escrito para el repo de GenAIOps con LangChain + FAISS +
> Streamlit. El enunciado permite explícitamente hacerlo "en Databricks o como lo estamos mostrando",
> así que esta entrega toma esa primera opción: **el mismo desafío, implementado sobre Databricks**.

---

## 1. Caso de uso

ServiCiudad atiende clientes de **energía, agua y gas**. El asistente responde dos tipos de pregunta:

1. **Datos del cliente** — contratos, facturación, estado de cuenta (funciones SQL en Unity Catalog).
2. **Documentación técnica y de políticas** — medidores, códigos de error, tarifas, reconexiones,
   PQR y emergencias (RAG sobre 8 manuales PDF indexados en Vector Search).

Datos sintéticos: 1.000 clientes, 1.534 contratos y 14.730 registros de facturación.

---

## 2. Cambios respecto al demo original

El demo original responde sobre una **empresa de telecomunicaciones**. Este fork lo lleva a servicios públicos. Todo lo
que cambió:

| Archivo | Cambio |
|---|---|
| `ai-agent/config.py` | `MODEL_NAME = asistente_servicios_publicos` |
| `ai-agent/_resources/02-data-generation.py` | Tipos de servicio `Energia` / `Agua` / `Gas` y sus tarifas base |
| `ai-agent/_resources/03-doc-pdf-documentation.py` | Los 8 manuales: medidor inteligente, códigos de error, PQR, tarifas residenciales, reconexión, prepago, acometidas y emergencias. Se agregó al prompt la restricción explícita de no hablar de telecomunicaciones |
| `ai-agent/_resources/01-setup.py` | Los datos y PDFs **siempre se generan localmente**; se eliminó la descarga del repositorio de dbdemos, que traía el corpus de telecomunicaciones |
| `ai-agent/_resources/04-eval-dataset-generation.py` | Dataset curado de **15 preguntas** en 4 tipos, con respuesta verificada contra los PDFs |
| `ai-agent/02-agent-eval/02.1_agent_evaluation.py` | System prompts de ServiCiudad |
| `ai-agent/03-knowledge-base-rag/03.1-pdf-rag-tool.py` | Prioridad de documentos, descripción del retriever y preguntas de ejemplo |
| `ai-agent/04-deploy-app/chatbot_app/main.py` | Título y textos de la app |
| `ai-agent/10-evaluadores-propios.py` | **Nuevo**: `@scorer` de código, juez `make_judge` y `Guidelines` del dominio |
| `ai-agent/11-acuerdo-humano.py` | **Nuevo**: etiquetado humano y acuerdo juez–humano |

Verificación de que no quedan restos del dominio anterior:

```bash
grep -ri "telecom\|router\|fiber\|adsl\|broadband" ai-agent/ --include="*.py" | grep -v "_resources/03-doc"
```

---

## 3. Cómo ejecutar

### Requisitos

- Databricks **Free Edition** (solo cómputo serverless)
- Versión de entorno **2 o superior** en el panel *Environment* de cada notebook
- Catálogo `workspace` (el de Free Edition)

### Orden

```
00 · ai-agent/config.py                     configuración (catálogo, esquema, modelos)
01 · ai-agent/01-create-tools/…             datos, PDFs y funciones de Unity Catalog
02 · ai-agent/02-agent-eval/02.1_…          agente, evaluación y despliegue del endpoint
03 · ai-agent/03-knowledge-base-rag/03.1_…  base de conocimiento con Vector Search
04 · ai-agent/04-deploy-app/04-…            chatbot en Databricks Apps
05 · ai-agent/05-production-monitoring/…    monitoreo con jueces registrados
10 · ai-agent/10-evaluadores-propios.py     evaluadores propios del dominio
11 · ai-agent/11-acuerdo-humano.py          acuerdo juez–humano
```

Los notebooks `06` y `07` (dashboard de negocio y MCP) son opcionales.

### Automatización

Los notebooks se pueden ejecutar por API con un job serverless:

```python
w.jobs.submit(
    tasks=[jobs.SubmitTask(
        task_key="nb",
        notebook_task=jobs.NotebookTask(notebook_path="<ruta del notebook>"),
        environment_key="Default",
    )],
    environments=[jobs.JobEnvironment(
        environment_key="Default",
        spec=compute.Environment(client="2"),
    )],
)
```

---

## 4. Mapeo a la rúbrica del desafío

| Parte | Puntos | Dónde está | Estado |
|---|---|---|---|
| 1 · Dominio | 10 | `config.py`, `_resources/02`, `_resources/03`, prompts | ✅ |
| 2 · Dataset | 15 | `_resources/04-eval-dataset-generation.py` | ✅ |
| 3 · Experimentos y diagnóstico | 20 | `02.1`, `03.1`, `docs/experimentos.md` | ⏳ |
| 4 · Evaluadores propios | 20 | `10-evaluadores-propios.py` | ⏳ |
| 5 · Confianza en el juez | 15 | `11-acuerdo-humano.py` | ⏳ |
| 6 · Producción | 10 | alias `model-to-deploy`, quality gate, CI | ⏳ |
| 7 · Informe | 10 | [`INFORME.md`](INFORME.md) | ⏳ |

---

## 5. Estructura

```
ai-agent/
├── config.py
├── 01-create-tools/01_create_first_billing_agent.py
├── 02-agent-eval/
│   ├── 02.1_agent_evaluation.py
│   └── agent.py
├── 03-knowledge-base-rag/03.1-pdf-rag-tool.py
├── 04-deploy-app/
├── 05-production-monitoring/
├── 10-evaluadores-propios.py
├── 11-acuerdo-humano.py
└── _resources/
docs/
├── experimentos.md          # tabla de configuraciones y diagnóstico de fallos
└── capturas/                # evidencia visual del informe
tests/
└── eval_dataset.json        # copia del dataset para revisión y CI
```

---

## 6. Créditos

Basado en la demo `ai-agent` de [`databricks-demos/dbdemos`](https://github.com/databricks-demos/dbdemos)
(Copyright Databricks, Inc.) a través de la adaptación en español para Free Edition de
[`darkanita/ai-agent-databricks`](https://github.com/darkanita/ai-agent-databricks).
Ver `ai-agent/_resources/LICENSE.py`.
