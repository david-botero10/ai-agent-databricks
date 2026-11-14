# Guion de la presentación (≈ 18 minutos)

Presentan: **David Botero**, **Jorge Giraldo** y **Samuel Padierna**. David comparte pantalla toda la sesión
(la demo necesita su sesión de Databricks). Diapositivas: 11 + demo en vivo.

**Antes de empezar (7:45):** abrir el notebook `12-demo-en-vivo` en Databricks y ejecutar **Run all** (~3 min).

---

## David — Problema, arquitectura y datos (≈ 5 min · diapositivas 1–4)

**1. Portada (30 s)**
> "Construimos un asistente para ServiCiudad, una empresa ficticia de agua, energía y gas. Pero el foco no es el chatbot:
> es poder demostrar con evidencia de MLflow cuál versión es mejor, por qué falla y cuánto confiar en la medición."

**2. El problema (1 min)**
> "Un cliente pregunta en la misma conversación por su factura y por su medidor. Lo primero está en una base de datos;
> lo segundo, en un manual PDF. Y hay un riesgo real: una mala respuesta sobre gas o electricidad puede causar un accidente."

**3. Arquitectura (1.5 min)**
> "El agente usa LangGraph y Llama 3.3 70B. Tiene dos herramientas: funciones de Unity Catalog para los datos del cliente
> y Vector Search sobre los 8 manuales. Todo queda trazado en MLflow 3, y cada cambio de prompt pasa por un quality gate
> en GitHub Actions que evalúa el agente en Databricks antes de fusionarse."

**4. Datos y documentos (1.5 min)**
> "1.000 clientes, 1.534 contratos y 14.730 facturas sintéticas, más 8 manuales. El demo original era de
> telecomunicaciones: eliminamos sus 39 manuales y barrimos todas las menciones en notebooks, prompts y app."

*Pase:* "Jorge les cuenta cómo medimos si el agente responde bien."

---

## Jorge — Cómo medimos y qué encontramos (≈ 6 min · diapositivas 5–8)

**5. Dataset (1.5 min)**
> "15 preguntas en cuatro tipos: 6 directas, 3 multi-documento, 3 sin respuesta y 3 adversarias. Cada respuesta
> esperada la verificamos contra su PDF. El demo traía preguntas sintéticas sin respuesta verificada que contaminaban el
> dataset: las sacamos."

**6. Evaluadores (1.5 min)**
> "Además de los jueces de MLflow, creamos cuatro propios. Dos scorers de código: uno detecta fugas de nombres internos y
> otro detecta teléfonos o correos inventados. Un juez con make_judge mide si un cliente entiende la respuesta, y una
> Guideline prohíbe indicar que se abra el medidor. Encontraron cosas que ningún juez integrado ve: un teléfono
> `1-800-SERVICIUAD` inventado y una fuga del nombre interno del índice."

**7. Experimentos (1.5 min)**
> "Cambiamos una variable a la vez. El RAG sube correctness de 0.40 a 0.53. k = 6 llega a 0.67 y es la única que responde
> una pregunta multi-documento. El chunking abarata pero destruye las directas: los manuales son cortos y partirlos separa
> la pregunta de su respuesta. Y somos honestos con el ruido: la misma configuración dio 0.53 y 0.67 en dos corridas."

**8. Diagnóstico de fallos (1.5 min)**
> "Leímos las trazas de cada fallo. Las multi-documento fallan porque el agente hace una sola búsqueda para dos temas y
> no le llega el segundo manual: eso es recuperación. Otros fallos son de generación: tenía el manual y omitió cifras.
> Y algunos no eran del agente: eran del dataset o del juez."

*Pase:* "Samuel les cuenta si podemos confiar en esos jueces y cómo llegó esto a producción."

---

## Samuel — Confianza, producción y cierre (≈ 5 min · diapositivas 9–11 + demo)

**9. ¿Confiamos en el juez? (1.5 min)**
> "Calificamos a mano 10 respuestas sin ver al juez y comparamos tres jueces. El gestionado coincidió en 0.80, GPT-OSS
> en 0.70 y Llama en 0.90, pero Llama es el mismo modelo del agente, así que se califica a sí mismo. Lo clave: todos los
> desacuerdos van en una dirección; ningún juez aprobó una respuesta mala. Elegimos el gestionado."

**10. Producción y CI (2 min)**
> "Promovimos la versión 3, con k = 6, con el alias model-to-deploy. El gate exige correctness 0.40 y safety 1.0: producción
> pasa con 0.667 y un prompt roto cae a 0.333 y el PR queda bloqueado. Y una lección: al principio el CI salía en verde
> con un log que decía NO CUMPLE, porque Databricks reintentaba solo la tarea fallida. Lo corregimos: si cualquier intento
> falla, el CI falla."

**Demo en vivo (2 min) — David opera la pantalla, Samuel narra**
1. "¿Qué significa el código de error 4001 de mi medidor ERR-MED-2024?" → responde con el manual (RAG).
2. "¿Quién es el gerente general de ServiCiudad?" → admite que no sabe, no inventa.
3. "Mi medidor está dañado, ¿puedo abrirlo y cambiar el cableado yo mismo?" → se niega y remite a un técnico.
> "Debajo de cada respuesta está su traza de MLflow: qué herramienta usó y qué documento recuperó."

**11. Cierre (1 min)**
> "La configuración de producción cuesta unos 11.800 tokens y 5.5 segundos por pregunta. La lección principal: medir
> antes de mejorar. Varios fallos eran del juez, del dataset o de la instrumentación; hasta el conteo de tokens venía
> inflado 60 veces. Con una semana más: búsqueda por subpreguntas para las multi-documento y un dataset de 50 preguntas."

---

## Preguntas probables

| Pregunta | Quién | Respuesta corta |
|---|---|---|
| ¿Por qué Databricks y no el repo local? | David | Lo acordamos con la profesora; usamos Unity Catalog, Vector Search y MLflow gestionado. |
| ¿k = 6 es realmente mejor? | Jorge | Es la mejor evidencia, no una prueba: +2 respuestas está en el borde del ruido. Por eso proponemos 50 preguntas. |
| ¿Por qué no hay endpoint desplegado? | David | En Free Edition la imagen del contenedor no terminó de construirse a tiempo, dos veces. La demo carga la misma versión desde Unity Catalog. |
| ¿Por qué el umbral es 0.40 y no 0.70? | Samuel | Producción midió 0.47–0.67 y una regresión grande 0.33: 0.40 es el punto medio. Con 0.70 fallaría `main`. |
| ¿Por qué no usaron RetrievalGroundedness? | Jorge | No puede leer el span del retriever, que guarda los documentos como texto. Clasificamos recuperación leyendo la traza. |
| ¿El primer prompt malo pasó el gate? | Samuel | Sí, con 0.467: estaba a 2 respuestas de producción. Con 15 preguntas el gate solo ve regresiones grandes; lo documentamos. |
