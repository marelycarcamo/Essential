# Decisión de personalización

## 1. El problema

Las respuestas del banco de prompts no se apoyan en ninguna fuente verificable:
ante la sigla PEP respondió con su significado en el ámbito financiero, y no hay
forma de auditar de dónde sale una afirmación ni de corregirla sin cambiar el
modelo.

## 2. Resultado del método de decisión

- RAG: 5 puntos
- PEFT: 1 punto
- Fine-tuning completo: 0 puntos

Ganadora única: **RAG**.

| Pregunta del método | Respuesta | Evidencia |
|---|---|---|
| El contenido cambia al menos una vez al mes | **sí** | Las 6 fuentes del corpus se crearon esta misma sesión y las 17 fuentes restantes del proyecto (13 módulos de `src/`, 2 de `docs/`, 2 de la raíz) cambian con cada bloque |
| Hay un problema de formato o de tono (≥15 % de 20 respuestas) | **no** | Se revisaron las 20 respuestas reales de `docs/resultados_metricas.json`: las 20 cumplen la instrucción de responder en una sola oración (0 % de incumplimiento) |
| Hay un dataset de calidad con 200 ejemplos etiquetados | **no** | Hay 13 ejemplos validados en `dataset_tuning.py`; el método pide 200 |
| El presupuesto es limitado o hay que iterar rápido | **sí** | No hay presupuesto para iterar entrenamientos y el plan gratuito no permite medir el efecto de un cambio en menos de un día |
| Se necesitan varias variantes del modelo | **no** | Un sistema, un idioma, una línea de producto |

## 3. Técnica elegida y por qué

La técnica elegida es **RAG**, aunque el puntaje no haya dependido de una sola
respuesta sino de tres que apuntan al mismo lado.

El problema es de **información y procedencia**, no de comportamiento: el modelo
sabe redactar, lo que no puede es demostrar de dónde saca lo que afirma. Un
ajuste de parámetros no agrega fuentes verificables; agrega ejemplos de
comportamiento, y aquí el comportamiento ya cumple —la revisión de las 20
respuestas dio 0 % de incumplimiento del formato—. Por eso quedan descartados
PEFT y fine-tuning completo, y no por falta de presupuesto: aunque hubiera
presupuesto, entrenar no resolvería el problema planteado.

El tercer punto que decide es el tiempo. Las fuentes cambian con cada bloque del
programa, así que cualquier conocimiento fijado en los pesos nace desactualizado:
el RAG indexa la documentación y la cita, y cuando el código cambia se reindexan
31 fragmentos en una sola llamada de embedding. El fine-tuning, en cambio,
quedaría obsoleto y habría que repetirlo.

**Corrección metodológica que conviene dejar por escrito:** la revisión
cualitativa del Bloque 17 atribuyó la baja precisión al incumplimiento de la
instrucción "responde en una sola oración". Medirlo la desmiente: las 20
respuestas son de una oración y la baja precisión se explica porque son más
largas que la referencia escrita a mano. La decisión se tomó sobre el número, no
sobre esa suposición anterior.

## 4. Qué necesitaría para ejecutarla

El índice existe y tiene **31 fragmentos de 6 documentos**, pero los seis son
documentación redactada a mano para este corpus. Quedan **17 fuentes reales sin
indexar**: 13 módulos de `src/` (el código que el sistema describe), 2 documentos
de `docs/` y 2 de la raíz. A esas 17 hay que agregar los tests, si se quiere que
el sistema pueda responder qué está cubierto por pruebas.

Trabajo estimado: 17 archivos a indexar, que con `indexar_corpus.py` son **una
corrida y una llamada de embedding**; el costo real de mantenerlo es de
redacción, no de cómputo, porque son archivos que existen.

Lo que **no** tengo y no necesito para RAG: un corpus de comentarios reales de
encuestas. Si más adelante se quisiera ir por PEFT, faltarían 187 ejemplos
etiquetados sobre los 13 actuales —unas 15,6 horas a 5 minutos por ejemplo— y esa
es precisamente la inversión que esta decisión descarta por ahora.
