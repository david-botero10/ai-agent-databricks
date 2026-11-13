# Guion de la presentación (15–20 min)

| # | Diapositiva | Min | Mensaje | Evidencia |
|---|---|---|---|---|
| 1 | ServiCiudad: el problema | 1 | Un asistente que responde sobre cuentas, facturas y manuales técnicos de agua, energía y gas | — |
| 2 | Arquitectura | 2 | Herramientas de Unity Catalog + RAG con Vector Search + LangGraph + MLflow 3 | Diagrama |
| 3 | Datos y documentos | 1 | 1.000 clientes, 14.730 facturas, 8 manuales; sin restos del dominio original | Captura de Unity Catalog |
| 4 | Dataset de evaluación | 2 | 15 preguntas en 4 tipos, cada respuesta verificada en el PDF | Tabla por tipo |
| 5 | Evaluadores | 2 | Jueces integrados + `@scorer` + `make_judge` + `Guidelines` del dominio | Traza con razonamiento |
| 6 | Experimentos | 3 | Prompt, k y chunking, de a una variable | Tabla de métricas + comparación de runs |
| 7 | Diagnóstico de fallos | 2 | 5 fallos clasificados con su traza | Tabla de fallos |
| 8 | ¿Confiamos en el juez? | 2 | Acuerdo juez–humano con dos jueces | Tabla de acuerdo |
| 9 | Producción y CI | 2 | Versión promovida, quality gate y PR bloqueado | Captura del CI fallando |
| 10 | Demo en vivo | 2 | Una pregunta directa, una sin respuesta y una adversaria | Review App / endpoint |
| 11 | Costo y conclusiones | 1 | Latencia y tokens; qué cambiaría con una semana más | — |

**Riesgo de la demo en vivo:** el endpoint de Free Edition escala a cero y tarda en despertar. Hacer una
consulta 10 minutos antes de presentar y tener capturas de respaldo.
