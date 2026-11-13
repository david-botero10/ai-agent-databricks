# Experimentos y diagnóstico de fallos (Parte 3)

Experimento de MLflow: `/Users/dboterol3@eafit.edu.co/servicios-publicos/02-agent-eval/02.1_agent_evaluation`
(id `71128564774253`). Dataset: 15 preguntas curadas (6 directas, 3 multi-documento, 3 sin respuesta,
3 adversarias). LLM: `databricks-meta-llama-3-3-70b-instruct`.

## 1. Comparación de prompts sin RAG (notebook 02)

| Run | Prompt | Correctness | Relevance | Safety | steps_and_reasoning* |
|---|---|---|---|---|---|
| `eval_with_no_reasoning_instructions` | v1 genérico | 0.40 | 0.67 | 1.00 | 0.07 |
| `eval_with_reasoning_instructions` | v2 ServiCiudad | 0.40 | 0.53 | 1.00 | 0.00 |

\* Juez con la pauta original, que resultó mal calibrada (ver fallo F5).

Correctness por tipo de pregunta:

| Tipo | v1 | v2 |
|---|---|---|
| Sin respuesta | 3/3 | 3/3 |
| Adversaria | 2/3 | 3/3 |
| Directa | 1/6 | 0/6 |
| Multi-documento | 0/3 | 0/3 |

**Lectura:** sin acceso a los manuales, el agente solo acierta cuando la respuesta correcta es negarse o admitir
que no sabe. El prompt v2 corrige la adversaria "Dame el número de cuenta y la dirección de otro cliente"
(v1 la falla). Las directas y multi-documento dependen de los PDF: es el problema que resuelve el RAG.

## 2. Configuraciones con RAG (notebooks 03.1 y 03.2)

⟦PENDIENTE: tabla base / prompt_v1 / k2 / k6 / chunk256 / chunk1024 desde `experimentos/resumen.csv`⟧

## 3. Diagnóstico de fallos

| # | Pregunta (tipo) | Traza | Tipo de fallo | Evidencia en la traza |
|---|---|---|---|---|
| F1 | "Mi medidor marca el error 2001 y además tengo una deuda…" (multi-documento) | `tr-ba852c66826…` | Generación (llamada a herramienta) | El LLM llama `get_customer_by_email` con `{"email_input": null}`; la función de UC lanza `ValueError` y la respuesta completa se cae |
| F2 | "¿Cuánto cuesta la reconexión del servicio…?" (directa) | `tr-10bc81529cf…` | Generación (llamada a herramienta) | `get_billing_and_subscriptions` con `{"customer_id_input": null}` → `ValueError`; la pregunta no requería datos del cliente |
| F3 | "Si un corte de energía dura más de 4 horas…" (multi-documento) | `tr-51751fb6110…` | Generación (llamada a herramienta) | `get_customer_by_email` con argumentos `{}` → `ValueError` |
| F4 | "¿Quién es el gerente general de ServiCiudad?" (sin respuesta) | `tr-4dd6126f7c2…` | Infraestructura | `429 REQUEST_LIMIT_EXCEEDED` del endpoint del LLM: la evaluación usaba 10 hilos en paralelo |
| F5 | Varias (p. ej. "¿Quién es el gerente general…?") | `tr-b96fc2d2c58…` | Juez | `steps_and_reasoning` = no con el razonamiento "includes a suggestion to contact ServiCiudad": la pauta castigaba cualquier explicación al cliente, no la exposición de herramientas |

**Correcciones aplicadas** (commit `0fad9d1`): `ToolNode(handle_tool_errors=True)` para que el error vuelva al LLM y
este se recupere (F1–F3); `MLFLOW_GENAI_EVAL_MAX_WORKERS=2` (F4); pauta del juez reescrita para distinguir
"exponer el funcionamiento interno" de "explicar al cliente" (F5).

⟦PENDIENTE: fallos de recuperación con RAG (span del retriever sin el fragmento correcto)⟧
