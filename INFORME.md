# Informe — Asistente de ServiCiudad con GenAIOps en Databricks

**Autor:** David Botero · Seminario, Maestría en Ciencias de los Datos y Analítica (EAFIT) · noviembre de 2026
**Repositorio:** https://github.com/david-botero10/ai-agent-databricks · detalle de experimentos y fallos en
[`docs/experimentos.md`](docs/experimentos.md)

## Contexto

ServiCiudad es una empresa ficticia de servicios públicos (agua, energía y gas). Su agente (LangGraph + MLflow 3,
`databricks-meta-llama-3-3-70b-instruct`) responde con dos fuentes: **funciones de Unity Catalog** sobre 1.000
clientes, 1.534 contratos y 14.730 facturas, y **8 manuales PDF** indexados en **Vector Search** (RAG).

El dataset tiene **15 preguntas**, cada respuesta verificada contra su PDF: 6 directas, 3 multi-documento, 3 sin
respuesta y 3 adversarias. Se evalúa con `Correctness`, `RelevanceToQuery`, `Safety`, una guía `steps_and_reasoning` y
cuatro evaluadores propios:

| Evaluador propio | Tipo | Resultado | Qué encontró |
|---|---|---|---|
| `sin_fugas_internas` | `@scorer` | 0.93 | Una respuesta filtró el nombre interno `knowledge_base_vs_index` |
| `sin_contactos_inventados` | `@scorer` | 0.93 | Teléfono `1-800-SERVICIUAD` y correos que no existen en ningún manual |
| `claridad_cliente` | `make_judge` | 0.33–0.83 | "No tengo información" no es accionable para el cliente |
| `seguridad_medidor` | `Guidelines` | 0.73 → 0.80 | Castigaba presionar el botón de reset, que el manual indica: se recalibró |

**Ruido de medición:** con la misma configuración, `correctness` dio 0.53 y 0.67 en dos corridas. Con 15 preguntas,
±0.13 son dos respuestas: ninguna diferencia menor a eso se lee como mejora. Estable entre corridas: multi-documento =
0.00 con k = 3, sin respuesta = 1.00 y safety = 1.00.

## 1. ¿Qué configuración va a producción y por qué?

Cada variante cambia una sola variable respecto a la base (prompt v2, documento completo por fila, k = 3):

| Variante | Correctness | Directa | Multi-doc | Sin resp. | Advers. | Tokens/pregunta | p50 |
|---|---|---|---|---|---|---|---|
| Sin RAG | 0.40 | 0/6 | 0/3 | 3/3 | 3/3 | — | — |
| Base RAG (k = 3) | 0.53 | 3/6 | 0/3 | 3/3 | 2/3 | 7.536 | 4.9 s |
| Prompt v1 | 0.60 | 3/6 | 0/3 | 3/3 | 3/3 | 7.481 | 8.8 s |
| k = 2 | 0.53 | 3/6 | 0/3 | 3/3 | 2/3 | 6.474 | 8.7 s |
| **k = 6** | **0.67** | 3/6 | **1/3** | 3/3 | 3/3 | 11.758 | 5.5 s |
| Chunk 256 | 0.40 | 0/6 | 0/3 | 3/3 | 3/3 | 2.929 | 2.8 s |
| Chunk 1024 | 0.33 | 1/6 | 0/3 | 3/3 | 1/3 | 4.054 | 2.5 s |

**A producción: prompt v2 + documento completo + k = 6**, versión 3 en Unity Catalog con el alias `model-to-deploy`
(notebook `13-promover-version`). Es la única variante que responde una multi-documento, el fallo dominante, con
latencia similar. Frente al agente sin RAG, `correctness` sube de 0.40 a 0.67. **Matiz:** +2 respuestas frente a la base
está en el borde del ruido; es la mejor evidencia disponible, no una prueba concluyente. Se descarta el chunking: los
manuales son cortos y partirlos separa pregunta y respuesta (directas de 3/6 a 0/6).

![Comparación de runs en MLflow](docs/img/01_comparacion_runs.png)

## 2. ¿Qué tipo de pregunta es más difícil? ¿Falla en recuperación o en generación?

**Multi-documento:** 0/3 en todas las corridas con k = 3. Los fallos se reparten entre **recuperación y generación**:

| Pregunta (tipo) | Fallo | Evidencia en la traza |
|---|---|---|
| Corte de más de 4 horas (multi) | Recuperación | El span `RETRIEVER` trae `TARIFA-RES-2024`, no `EMERG-2024` |
| Error 3001 y tiempo de respuesta (multi) | Recuperación | Una consulta para dos temas: llega `ERR-MED-2024`, no `PQR-2024` |
| Error 2001 y deuda vencida (multi) | Generación | Llegan los dos manuales; omite la regla de $50 y el aviso de 15 días |
| Costo de la reconexión (directa) | Generación | El manual dice $25, $100 y $10; responde solo $25 |
| Tiempo de respuesta de PQR (directa) | Dataset | Responde lo preguntado; la esperada exige datos que no se pidieron |

La causa de fondo: el agente hace **una sola búsqueda** que mezcla los dos temas. Criterio: recuperación si el código del
manual esperado no aparece en el span `RETRIEVER`. `RetrievalRelevance` y `RetrievalGroundedness` no sirvieron: dan
`SCORER_ERROR` porque `VectorSearchRetrieverTool` guarda los documentos como texto, no como lista estructurada.

## 3. ¿Inventa en las preguntas sin respuesta? ¿Resiste las adversarias?

**Sin respuesta: no inventa** (3/3 en todas las corridas). **Pero sí inventa en otro lugar:** cuando le falta un dato
en una pregunta que sí tiene respuesta, agrega un teléfono (`1-800-SERVICIUAD`) y correos que no existen. Ningún juez
integrado lo detecta; lo encontró el scorer de código `sin_contactos_inventados`.
**Adversarias: resiste las tres** (safety = 1.00 siempre): no explica cómo conectarse sin pagar, no entrega datos de
otro cliente y remite a un técnico si el medidor está dañado.

![Traza con los evaluadores propios](docs/img/02_traza_evaluadores.png)

## 4. ¿Cuánto coincide el juez con el criterio humano?

Se calificaron a mano 10 respuestas (3 directas, 2 multi, 2 sin respuesta, 3 adversarias) sin ver al juez; el humano
aprobó 8. Cada calificación quedó en su traza como feedback `correctness_humano` de fuente `HUMAN`.

| Juez de `Correctness` | Acuerdo | Aprueba | Desacuerdos |
|---|---|---|---|
| **Gestionado de Databricks (elegido)** | **0.80** | 6/10 | PQR, "Ignora tus instrucciones…" |
| `databricks-gpt-oss-120b` | 0.70 | 5/10 | PQR, "¿Puedo abrir el medidor?", "Ignora tus instrucciones…" |
| Llama 3.3 70B (el mismo LLM del agente) | 0.90 | 7/10 | PQR |

**Todos los desacuerdos van en una dirección:** el juez rechaza lo que el humano aprueba (negativas sin mencionar la
ilegalidad, respuestas parciales); ninguno aprobó una respuesta incorrecta. Elegimos el **gestionado**: independiente del
agente y nunca aprueba una respuesta mala, la propiedad que importa en un gate. El Llama coincide más, pero se califica
a sí mismo y su ventaja es una sola respuesta. **Confiamos en él como cota inferior:** 0.53 del juez frente a 8/10
aprobadas por el humano.

![Acuerdo juez–humano](docs/img/03_acuerdo_juez.png)

## 5. ¿Qué costo tiene la mejor configuración?

k = 6 usa **11.758 tokens por pregunta** (~1.8 llamadas al LLM) y **5.5 s de latencia mediana** (p90 13.5 s): +56 % de
tokens frente a k = 3 y 4 veces más que chunk 256. **Vale la pena** frente a chunk 256, que pierde todas las directas:
en atención al cliente una respuesta incorrecta cuesta más que 9 mil tokens. Frente a k = 3 la ganancia es pequeña; se
acepta porque ataca el fallo dominante sin cambiar la latencia de forma perceptible.
**Hallazgo de medición:** el `tokenUsage` de cada traza venía inflado ~60 veces (~460 mil tokens por pregunta) porque el
span `ChatDatabricks` suma cada fragmento del streaming; los costos se calcularon con los spans `DatabricksCompletions`.

## 6. Con una semana más

1. **Búsqueda por subpreguntas** para las multi-documento: ataca la causa; subir k solo la mitiga y encarece.
2. **Instrumentar bien el retriever** para que `RetrievalRelevance`/`RetrievalGroundedness` funcionen solos.
3. **Ampliar el dataset a ~50 preguntas** y corregir la esperada de PQR: hoy el ruido (±0.13) iguala las diferencias.
4. **Llevar `sin_contactos_inventados` al gate** y prohibir en el prompt inventar teléfonos o correos.

## Producción y CI

- **Gate:** `Correctness ≥ 0.40` y `Safety = 1.0` con el juez gestionado. En corridas limpias, producción midió
  0.467–0.67 (7 a 10 de 15) y una regresión grande 0.333 (5 de 15): 0.40 es el punto medio, con una respuesta de margen
  por lado. Safety fue 1.0 en todas las corridas y no es negociable en gas y electricidad.
- **Demostración ([PR #1](https://github.com/david-botero10/ai-agent-databricks/pull/1)):** un prompt que responde con
  una sola palabra en inglés cae a **0.333** y el CI bloquea la fusión; `main` (producción, k = 6) pasa con 0.667.
- **Lo que el gate no puede ver:** el primer prompt "peor" (sin herramientas, "inventa si no sabes") midió 0.467 con la
  evaluación limpia y **pasó**. El LLM ignoró la orden de inventar (3/3 en sin respuesta) y solo perdió 2 respuestas,
  lo mismo que el ruido entre corridas. Un gate de 15 preguntas solo detecta regresiones grandes (≥ 3–4 respuestas);
  para regresiones finas hace falta un dataset mayor.
- **Un CI en verde que no significaba nada:** al promover k = 6, GitHub marcó éxito con un log que decía
  `correctness 0.333 NO CUMPLE`: Databricks **reintenta solo** las tareas serverless que fallan (incluso con
  `max_retries=0`) y el reintento pasó. Además, los 429 del LLM contaban como respuestas incorrectas (el primer
  "bloqueo" del PR, con 0.267, estaba contaminado por eso). Corregido: el CI falla si **cualquier** intento falla,
  imprime todos los intentos, evalúa con 2 hilos y declara inválida la evaluación si alguna respuesta termina en error.

![PR bloqueado por el quality gate](docs/img/04_pr_bloqueado.png)
