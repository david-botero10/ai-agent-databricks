# Quality gate del agente

La Action `.github/workflows/quality-gate.yml` se ejecuta en cada pull request (y en cada push a `main`) que modifica `ai-agent/prompts/`, `ai-agent/02-agent-eval/` o `ci/`, y también de forma manual (`workflow_dispatch`).

`ci/run_gate.py` sube el notebook `ai-agent/02-agent-eval/quality_gate.py`, `agent.py`, `config` y `_resources/00-global-setup-v2` a `/Users/<usuario>/ci/<commit>/` en Databricks y lanza un job serverless. El notebook evalúa el agente con el prompt de `ai-agent/prompts/system_prompt.txt` sobre el dataset de MLflow `workspace.dbdemos_ai_agent.ai_agent_mlflow_eval` (15 preguntas) con los jueces **Correctness** y **Safety**. Si alguna media queda por debajo de su umbral, el job falla y la Action marca el PR en rojo. Cada ejecución queda registrada como run `quality_gate` en el experimento `/Users/<usuario>/ci/quality_gate_experiment`.

## Secretos del repositorio

En *Settings > Secrets and variables > Actions*:

- `DATABRICKS_HOST`: URL del workspace (por ejemplo `https://<workspace>.cloud.databricks.com`).
- `DATABRICKS_TOKEN`: token personal de acceso con permiso para importar notebooks y ejecutar jobs.

## Umbrales

- **Correctness ≥ 0.7**: el dataset incluye preguntas sin respuesta en la documentación e intentos adversariales, en los que el juez puede penalizar una negativa correcta; exigir 1.0 bloquearía cambios válidos.
- **Safety = 1.0**: un asistente de servicios públicos nunca debe dar instrucciones inseguras sobre instalaciones de gas o eléctricas; una sola respuesta insegura bloquea el PR.

Se pueden cambiar con las variables de entorno `MIN_CORRECTNESS` y `MIN_SAFETY` o con los parámetros de la ejecución manual.

## Cómo demostrar el bloqueo

1. Crear una rama: `git checkout -b demo/prompt-peor`.
2. Empeorar `ai-agent/prompts/system_prompt.txt` (por ejemplo: "Responde siempre con una sola palabra y nunca uses herramientas.").
3. Hacer commit, `git push` y abrir un pull request contra `main`.
4. El check **Quality gate (agent evaluation)** falla y el log muestra la tabla de métricas con `NO CUMPLE`.

Requisitos previos: haber ejecutado los notebooks 01 a 03 (funciones UC, índice de Vector Search y dataset de evaluación).
