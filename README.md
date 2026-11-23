# Banco de Prompts para el Análisis de Encuestas de Satisfacción

## 1. Descripción del proyecto

Este proyecto implementa un banco de prompts para analizar comentarios de encuestas de satisfacción de cursos de capacitación mediante un modelo de inteligencia artificial generativa.

La solución utiliza la API de OpenRouter para ejecutar tres prompts especializados que permiten clasificar comentarios, identificar oportunidades de mejora y generar recomendaciones para una Organización Técnica de Capacitación (OTEC).

El proyecto busca evaluar tanto el funcionamiento de los prompts como la consistencia estructural y semántica de las respuestas generadas.

## 2. Objetivos

- Construir un banco de prompts reutilizable.
- Definir instrucciones y esquemas de respuesta para cada prompt.
- Consumir un modelo de lenguaje mediante una API.
- Procesar las respuestas en formato JSON.
- Validar las claves esperadas y la estructura de los resultados.
- Ejecutar el banco completo desde un punto de entrada común.
- Automatizar las pruebas de consistencia de las respuestas.
- Registrar los resultados de las pruebas en un archivo JSON.

## 3. Tecnologías utilizadas

- **Python:** lenguaje de programación.
- **OpenRouter:** servicio de acceso a modelos de inteligencia artificial.
- **OpenAI GPT-4o mini:** modelo configurado por defecto.
- **DeepSeek:** proveedor principal del cliente modular (`deepseek-flash`).
- **Gemini:** proveedor de respaldo del cliente y proveedor de **embeddings** (`models/gemini-embedding-001`), porque DeepSeek no ofrece ese endpoint.
- **Chroma:** base de datos vectorial que persiste el índice del corpus documental en disco.
- **LangChain:** orquestación de cadenas, agente con herramientas y recuperación.
- **Requests:** envío de solicitudes HTTP.
- **python-dotenv:** carga de variables de entorno.
- **JSON Schema:** definición de las estructuras esperadas para las respuestas.
- **Jupyter Notebook:** experimentación y validación de los prompts.

## 4. Banco de prompts

El proyecto contiene tres prompts especializados.

### 4.1. Clasificar comentarios

Identifica las dimensiones de la capacitación mencionadas por los alumnos y clasifica sus valoraciones.

Las dimensiones contempladas incluyen facilitación, contenidos, materiales, infraestructura, coordinación, tiempo práctico, alimentación y valoración general.

Las valoraciones posibles son positiva, negativa, mixta y neutral.

### 4.2. Identificar oportunidades

Analiza comentarios de alumnos para identificar:

- Fortalezas de la capacitación.
- Oportunidades de mejora.

El prompt agrupa comentarios que expresan ideas similares y evita inventar opiniones o atribuir recurrencia sin evidencia suficiente.

### 4.3. Generar recomendaciones

Propone acciones concretas y aplicables para abordar las oportunidades de mejora identificadas.

Cada recomendación incluye una dimensión, una acción y un nivel de prioridad: alta, media o baja.

## 5. Estructura del proyecto

```text
Essential/
├── corpus/
│   ├── 01_cliente_modular.md
│   ├── 02_costos.md
│   ├── 03_banco_prompts.md
│   ├── 04_agente_herramientas.md
│   ├── 05_pruebas.md
│   └── 06_busqueda_pipeline.md
├── docs/
│   ├── decision-arquitectura.md
│   ├── Diagrama de flujo para consumir OpenRouter.png
│   ├── Evidencia_peticion_autenticada_Thunder_Client.png
│   ├── Evidencia_peticion_evidencia_autenticada_postman.png
│   ├── limites_metrica.md
│   ├── registro_pipeline_texto.txt
│   ├── resultados_metricas.json
│   └── revision_manual.json
├── notebooks/
│   └── prueba_banco_prompts.ipynb
├── prompts/
│   ├── banco.py
│   ├── registro_resultados.json
│   └── test_consistencia.py
├── src/
│   ├── banco_prompts.py
│   ├── cliente.py
│   ├── comparar_modelos.py
│   ├── consumir_deepseek.py
│   ├── consumir_gemini.py
│   ├── consumir_openrouter.py
│   ├── costos.py
│   ├── errores.py
│   ├── metricas.py
│   ├── parsear_json.py
│   └── pipeline_texto.py
├── tests/
│   ├── conftest.py
│   ├── test_banco_prompts.py
│   ├── test_cliente.py
│   ├── test_consumir_deepseek.py
│   ├── test_consumir_gemini.py
│   ├── test_consumir_openrouter.py
│   ├── test_costos.py
│   ├── test_integracion.py
│   ├── test_main.py
│   ├── test_metricas.py
│   ├── test_parsear_json.py
│   └── test_pipeline_texto.py
├── experimentos/
│   ├── ensayo_rate_limit.py
│   ├── listar_fragmentos.py
│   ├── medir_metricas.py
│   ├── medir_pipeline.py
│   └── verificar_indice.py
├── .env.example
├── .gitignore
├── evaluacion_recuperacion.py
├── indexar_corpus.py
├── main.py
├── pytest.ini
├── recuperacion.py
├── README.md
├── requirements.txt
├── resultados_evaluacion.txt
├── resultados_evaluacion_k5.txt
├── revision_manual.py
├── test_evaluacion_recuperacion.py
├── test_indexar_corpus.py
└── test_recuperacion.py

```

**Nota:** La estructura muestra los archivos principales del proyecto. Las carpetas `docs/` y `tests/` pueden contener archivos adicionales. El archivo `prompts/registro_resultados.json` se genera al ejecutar las pruebas de consistencia.

### Responsabilidades de los archivos principales

| Archivo | Responsabilidad |
|---|---|
| `main.py` | Coordina la ejecución del banco de prompts y presenta un resumen. |
| `src/banco_prompts.py` | Define el banco utilizado por el punto de entrada principal. |
| `src/consumir_openrouter.py` | Prepara las solicitudes, consulta la API y extrae las respuestas. |
| `src/consumir_gemini.py` | Proveedor de respaldo: arma, envía y extrae la respuesta de la API de Gemini con el mismo contrato de llamada que DeepSeek. |
| `src/cliente.py` | Expone `consultar_modelo()` como única función pública: arma la petición, la envía con reintento automático y devuelve el texto. |
| `src/errores.py` | Define las excepciones propias del cliente y su relación con las excepciones estándar de Python, incluida una específica para el límite de uso del proveedor. |
| `src/costos.py` | Calcula el costo de cada operación a partir de los tokens informados por el proveedor, con tarifas por componente y distinción entre hora punta y hora valle, y mantiene el acumulado de la sesión. |
| `src/parsear_json.py` | Procesa las respuestas JSON y comprueba las claves esperadas. |
| `tests/conftest.py` | Agrega `src/` al `sys.path` y reinicia el acumulado de costo antes y después de cada prueba. |
| `pytest.ini` | Excluye por defecto el test que llama al proveedor real, para que la suite no consuma cuota. |
| `experimentos/ensayo_rate_limit.py` | Provoca un límite de uso real contra el proveedor de respaldo y muestra el backoff y su techo en vivo. |
| `src/pipeline_texto.py` | Normaliza y tokeniza texto antes de medirlo: minúsculas, sin puntuación y con espacios simples. Concentra la limpieza y la tokenización en una sola función para que no se aplique una y se olvide la otra. |
| `src/metricas.py` | Calcula precisión, cobertura y F1 comparando dos textos ya normalizados. Cuenta repeticiones con `Counter` y toma el mínimo entre ambos lados, así que una respuesta repetitiva no infla el puntaje. |
| `experimentos/medir_metricas.py` | Genera una respuesta real por consulta, calcula las tres métricas y guarda los resultados con la respuesta incluida. |
| `docs/resultados_metricas.json` | Resultado de esa corrida: consulta, respuesta y las tres métricas de cada caso. |
| `revision_manual.py` | Registro de la revisión cualitativa de 10 casos, con la rúbrica de tres preguntas aplicada a cada uno y su justificación escrita. |
| `docs/revision_manual.json` | El mismo registro en JSON, generado por `revision_manual.py`. |
| `docs/limites_metrica.md` | Documento de límites de la métrica: qué mide, modos de falla con un ejemplo concreto de cada uno, muestra revisada y recomendación de uso. |
| `experimentos/medir_pipeline.py` | Genera tres respuestas reales con el cliente y calcula la coincidencia contra una referencia escrita a mano, sobre texto crudo y sobre texto procesado. |
| `docs/registro_pipeline_texto.txt` | Registro de esa corrida: consulta, referencia, respuesta, las dos proporciones y el costo acumulado. |
| `corpus/` | Corpus documental del proyecto en seis documentos cortos: cliente, costos, banco de prompts, agente, pruebas y búsqueda. Es la fuente que se indexa para la recuperación semántica, y cada fragmento conserva el nombre del archivo del que salió para poder citarlo. |
| `indexar_corpus.py` | Lee `corpus/`, lo parte en fragmentos y guarda el índice con embeddings reales en `vectorstore/`. Verifica la integridad de los fragmentos antes de gastar llamadas y deja registrado el modelo con el que se construyó el índice. |
| `experimentos/verificar_indice.py` | Se reconecta al índice desde un proceso nuevo, comprueba el conteo y la metadata de la colección, y hace tres búsquedas por significado para evidenciar que el salto semántico funciona. |
| `test_indexar_corpus.py` | Pruebas de la indexación con embeddings falsos: carga, partición, ids, integridad y persistencia real en un directorio temporal, sin red ni cuota. |
| `recuperacion.py` | Recupera por significado los fragmentos más parecidos a una pregunta, los numera y arma el prompt que obliga a citar la fuente. Devuelve el mapa de citas y los problemas detectados en ellas. |
| `test_recuperacion.py` | Pruebas de la recuperación con un índice real en directorio temporal y un modelo de generación doble: numeración, `k`, validación de la pregunta y detección de citas inventadas. |
| `evaluacion_recuperacion.py` | Corre 15 preguntas etiquetadas a mano contra el retriever real y calcula precisión y recall por caso y promedio. Guarda la corrida sin sobrescribir las anteriores y lee la configuración del índice en vez de escribirla a mano. |
| `resultados_evaluacion.txt` | Corrida de referencia con `k=3`: una línea de fecha, una de configuración, una por pregunta y el promedio. |
| `resultados_evaluacion_k5.txt` | La misma evaluación con `k=5`, para comparar el trade-off de precisión y recall entre las dos configuraciones. |
| `experimentos/listar_fragmentos.py` | Imprime cada fragmento del índice con su id. Es lo que se usa para etiquetar a mano el conjunto de evaluación leyendo el corpus, no el retriever. |
| `test_evaluacion_recuperacion.py` | Pruebas de las fórmulas, del promedio con casos sin respuesta, del formato del archivo y de la negativa a sobrescribir una corrida anterior. |
| `prompts/banco.py` | Define los prompts y sus esquemas para las pruebas de consistencia. |
| `prompts/test_consistencia.py` | Ejecuta los prompts varias veces, valida las respuestas y registra los resultados. |
| `prompts/registro_resultados.json` | Almacena los resultados de la última ejecución de las pruebas de consistencia. |
| `notebooks/prueba_banco_prompts.ipynb` | Permite experimentar y evaluar los prompts durante el desarrollo. |

Los archivos `src/banco_prompts.py` y `prompts/banco.py` son independientes. Actualmente, `main.py` utiliza el primero, mientras que el script de consistencia utiliza el segundo.



## 6. Requisitos previos

Se necesita:

- Python instalado.
- Al menos una clave API del proveedor que se vaya a usar: **OpenRouter** para `main.py` y el script de consistencia, **DeepSeek** para el cliente modular. Gemini es opcional, como respaldo.
- Las dependencias declaradas en `requirements.txt`.

Instala las dependencias desde la raíz del repositorio:

```bash
pip install -r requirements.txt
```

Si trabajas con un entorno virtual, actívalo antes de ejecutar los comandos.

### 6.1. Entorno virtual y elección del intérprete

El proyecto se ejecuta dentro de un entorno virtual propio. Actívalo antes de instalar dependencias o de ejecutar cualquier comando:

```powershell
& .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

En macOS o Linux el comando equivalente es `source .venv/bin/activate`.

Una vez activado, `python`, `pip` y `pytest` apuntan al entorno del proyecto sin necesidad de escribir la ruta completa.

**No uses el lanzador `py` de Windows.** Ese lanzador ejecuta el Python instalado en el sistema, no el del proyecto. Los dos pueden tener exactamente la misma versión, así que el cambio de intérprete no produce ningún mensaje de error: solo aparecen librerías faltantes, o se instalan en un entorno que no corresponde.

Por la misma razón, `pip freeze > requirements.txt` solo debe ejecutarse con el entorno virtual activo. Ese archivo describe las dependencias del proyecto, no las de la máquina.

Como comprobación rápida, el intérprete en uso debe apuntar a la carpeta `.venv`:

```powershell
python -c "import sys; print(sys.executable)"
```

## 7. Configuración

Crea un archivo `.env` en la raíz del proyecto. El archivo `.env.example` lista todas las variables con valores de referencia; esta es la misma lista:

```dotenv
# OpenRouter: lo usan main.py y prompts/test_consistencia.py
OPENROUTER_API_KEY=tu_clave_api
OPENROUTER_MODEL=openai/gpt-4o-mini
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1/chat/completions

# DeepSeek: proveedor principal del cliente modular
DEEPSEEK_API_KEY=tu_clave_deepseek
DEEPSEEK_MODEL=deepseek-flash
DEEPSEEK_BASE_URL=https://api.deepseek.com

# Gemini: proveedor de respaldo del cliente modular
GEMINI_API_KEY=tu_clave_gemini
GEMINI_MODEL=gemini-3.8-flash
GEMINI_BASE_URL=https://generativelanguage.googleapis.com/v1beta
```

Reemplaza cada valor por tu propia clave. Solo necesitas las del proveedor que vayas a usar: `main.py` no lee las de DeepSeek, y el cliente modular no lee las de OpenRouter.

`GEMINI_MODEL` tiene que ser un nombre que exista en el catálogo de la API. Si no lo es, la llamada devuelve **HTTP 404** con el mensaje `models/<nombre> is not found`. El catálogo actual se consulta en `GET https://generativelanguage.googleapis.com/v1beta/models`.

**Importante:** No compartas tu clave API ni la subas al repositorio. El archivo `.env` debe estar excluido mediante `.gitignore`. El archivo `.env.example` puede documentar las variables necesarias sin incluir credenciales reales.


### 7.1 Diagrama de flujo para consumir OpenRouter
![Diagrama técnico e instructivo titulado Diagrama de flujo – Consumir OpenRouter. Sobre fondo blanco, cajas de colores y flechas muestran cómo una pregunta se envía a OpenRouter, se extrae el texto de la respuesta y se calcula el costo. El flujo comienza en Inicio y obtener_dato_de_prueba(pregunta), continúa con armar_peticion(pregunta), enviar_peticion(url, headers, body) y extraer_texto(datos), y termina con el resultado para quien llamó la función. La petición valida que la pregunta no esté vacía, obtiene OPENROUTER_API_KEY desde .env mediante os.getenv() y construye URL, headers de Authorization y body. La respuesta HTTP se valida y se convierte con respuesta.json(); después se extrae choices[0].message.content. Un proceso adicional usa prompt_tokens y completion_tokens para calcular el costo de la consulta en USD según los precios del modelo. El panel Manejo de errores enumera ValueError para pregunta vacía y RuntimeError para clave ausente, conexión, timeout, error HTTP, JSON inválido o estructura inesperada; el bloque principal captura ValueError y RuntimeError y muestra No se pudo completar la operación. También aparecen ConnectionError, Timeout, JSONDecodeError, KeyError, IndexError y TypeError. La leyenda distingue función, proceso, decisión, error o excepción, cálculo, flujo normal y flujo de error. Las flechas identifican pregunta de entrada, petición, respuesta HTTP cruda, JSON, texto generado, datos de uso y costo calculado. Una sección final lista GEMINI_API_KEY / OPENROUTER_API_KEY, manejo de errores, cálculo de costo y el orden de funciones armar_peticion → enviar_peticion → extraer_texto.](docs/Diagrama_de_flujo_para_consumir_OpenRouter.png)

## 8. Ejecución del proyecto

### 8.1. Ejecutar el banco de prompts

Desde la raíz del repositorio, ejecuta:

```bash
python main.py
```

El programa:

1. Carga la configuración.
2. Recupera el banco de prompts.
3. Envía las solicitudes a OpenRouter.
4. Procesa las respuestas JSON.
5. Comprueba las claves esperadas.
6. Informa el resultado de cada ejecución.
7. Presenta un resumen con el total de prompts ejecutados, los éxitos, los fallos y el porcentaje de éxito.

Cada ejecución realiza nuevas solicitudes al modelo y puede consumir cuota o créditos de la API.

### 8.2. Ejecutar las pruebas de consistencia

Desde la raíz del repositorio, ejecuta:

```bash
python prompts/test_consistencia.py
```

El script ejecuta cada prompt tres veces por defecto y valida las respuestas recibidas.

Durante la ejecución:

1. Envía las solicitudes al modelo configurado.
2. Procesa las respuestas en formato JSON.
3. Comprueba la estructura de los resultados.
4. Valida los tipos de datos y los valores permitidos definidos en los esquemas.
5. Registra los intentos fallidos y sus motivos.
6. Calcula el porcentaje de respuestas válidas.
7. Genera o actualiza `prompts/registro_resultados.json`.

Se incluye una pausa entre solicitudes para espaciar las llamadas a la API y reducir el riesgo de alcanzar los límites de cuota.

### 8.3. Ejecutar las pruebas automatizadas

Desde la raíz del repositorio:

```bash
python -m pytest
```

La suite **no toca la red**: las llamadas al proveedor se reemplazan por dobles y `time.sleep` se intercepta, así que no consume cuota y corre en menos de un segundo. El único test que llama al proveedor real está marcado y excluido por defecto; para correrlo a propósito: `python -m pytest -m integracion`.

### 8.4. Reproducir el ensayo de rate limit

```bash
python experimentos/ensayo_rate_limit.py
```

Provoca un límite de uso real contra Gemini y muestra cómo el backoff espera cada vez más hasta llegar a su techo. **Consume cuota del proveedor**, así que no forma parte de la suite automática.

### 8.5. Medir una respuesta antes y después del pipeline

```bash
python experimentos/medir_pipeline.py
```

Genera tres respuestas reales con el cliente, las compara contra una referencia escrita a mano y muestra la misma métrica calculada sobre el texto crudo y sobre el texto procesado. **Consume tres llamadas al proveedor principal** y deja el registro en `docs/registro_pipeline_texto.txt`.

### 8.6. Medir respuestas con precisión, cobertura y F1

```bash
python experimentos/medir_metricas.py
```

Genera una respuesta por consulta, calcula las tres métricas y guarda `docs/resultados_metricas.json` **con la respuesta incluida**, para poder releer los casos bajos sin regenerarlos. Consume una llamada al proveedor por cada caso.

### 8.7. Indexar el corpus documental

```bash
python indexar_corpus.py
```

Lee los seis documentos de `corpus/`, los parte en fragmentos, calcula un embedding real por fragmento y los persiste en `vectorstore/`. Deja impreso el número de documentos, el de fragmentos y la metadata de la colección. Se corre **una vez**, o cada vez que cambie el corpus: la recuperación se reconecta al índice y no vuelve a calcular embeddings.

`vectorstore/` está en `.gitignore` porque es contenido generado y se puede reconstruir con este comando. Para comprobar que el índice quedó bien, `python experimentos/verificar_indice.py`.

### 8.8. Consultar el flujo de recuperación

```bash
python recuperacion.py "¿qué pasa si la llamada al proveedor falla varias veces?"
```

Recupera los tres fragmentos más parecidos por significado, arma el prompt con el contexto numerado, genera la respuesta y muestra el mapa de citas. Requiere el índice creado con el comando anterior.

### 8.9. Evaluar la recuperación con métricas

```bash
python evaluacion_recuperacion.py
python evaluacion_recuperacion.py --k 5 --salida resultados_evaluacion_k5.txt
```

Corre las 15 preguntas del conjunto de evaluación contra el retriever real y guarda precisión@k y recall@k por pregunta, más el promedio. Se niega a sobrescribir un archivo existente: cada configuración nueva va a su propio archivo, porque la comparación es lo que da valor a la medición.

Para etiquetar un caso nuevo o reetiquetar cuando cambie el corpus, `python experimentos/listar_fragmentos.py` imprime cada fragmento con su id.

## 9. Pruebas y validación

El proyecto contempla distintas instancias de validación:

- Exploración y experimentación mediante el notebook.
- Ejecución del banco completo mediante `main.py`.
- Validación automatizada de las respuestas mediante `prompts/test_consistencia.py`.
- Registro estructurado de los resultados en formato JSON.

### 9.1. Resultados de ejecución de `main.py`

En la ejecución realizada durante el desarrollo se obtuvo:

| Indicador | Resultado |
|---|---:|
| Prompts ejecutados | 3 |
| Ejecuciones correctas | 3 |
| Ejecuciones fallidas | 0 |
| Porcentaje de éxito | 100 % |

### 9.2. Resultados de las pruebas de consistencia

En la última ejecución de `prompts/test_consistencia.py` se obtuvieron los siguientes resultados:

| Prompt | Intentos | Respuestas válidas | Porcentaje |
|---|---:|---:|---:|
| `clasificar_comentario` | 3 | 3 | 100 % |
| `identificar_oportunidades` | 3 | 3 | 100 % |
| `generar_recomendaciones` | 3 | 3 | 100 % |
| **Total** | **9** | **9** | **100 %** |

El archivo `prompts/registro_resultados.json` almacena los resultados de esta ejecución, incluyendo los intentos realizados, los éxitos y los fallos registrados.

Estos resultados corresponden a ejecuciones específicas durante el desarrollo. No garantizan que futuras respuestas del modelo tengan el mismo comportamiento.

### 9.3. Consistencia estructural y semántica

La validación estructural comprueba que las respuestas cumplan los requisitos definidos en los esquemas, como la presencia de claves obligatorias, los tipos de datos y los valores permitidos.

Sin embargo, una respuesta estructuralmente válida puede contener interpretaciones discutibles o recomendaciones poco pertinentes. Por eso, la validación estructural no sustituye la evaluación semántica del contenido.

Para evaluar la consistencia de forma más amplia, es necesario incorporar casos de prueba adicionales y criterios de evaluación definidos.

### 9.4. Verificación del cliente modular

La entrega del Bloque 12 se verificó en dos niveles: pruebas automatizadas, que no realizan llamadas reales, y escenarios manuales contra la API.

**Pruebas automatizadas.** La suite reúne 115 pruebas. Las que ejercitan el cliente reemplazan la llamada de red por un doble y sustituyen `time.sleep`, de modo que los reintentos se prueban sin esperar ni consumir cuota.

**Escenarios manuales.**

| # | Escenario | Cómo se ejecutó | Resultado |
|---|---|---|---|
| 1 | Consulta normal: el cliente devuelve el texto del modelo | Pendiente de ejecución con la API real | — |
| 2 | Credenciales inválidas: se clasifica como error de configuración y no se reintenta | Llamada real con clave inválida, sobre DeepSeek y sobre Gemini | Correcto en ambos proveedores |
| 3 | Falla temporal (429 o 5xx): se reintenta antes de fallar | Dobles de red en la suite | Correcto: tres intentos y luego error propio |
| 4 | Reintentos agotados: el error final es explícito y conserva la causa original | Dobles de red en la suite | Correcto: se levanta `ErrorReintentosAgotados` |
| 5 | Entrada no textual: el cliente informa el problema en lugar de fallar con un error interno | Ejecución local con una lista en lugar de un texto | Correcto: `ValueError` con mensaje explícito |

**Sobre la clasificación en Gemini.** Google no responde 401 cuando la clave es inválida: devuelve 400 con el campo `error.details[].reason` en `API_KEY_INVALID`. La detección se implementó leyendo ese campo estructurado y no el texto del mensaje, porque la redacción del texto puede cambiar. El caso se verificó contra la API real.

Los escenarios 3 y 4 se validan con dobles de red porque una falla temporal no se puede provocar a voluntad sobre el proveedor. La limitación se asume de forma explícita: los dobles confirman que el cliente reacciona como corresponde, no que el proveedor responda 429 en ese momento.

### 9.5. Verificación del proveedor de respaldo

`src/consumir_gemini.py` existe desde el Bloque 12 como respaldo del proveedor principal, pero hasta el Bloque 13 nunca se había ejecutado contra la API real.

Al ejercitarlo por primera vez devolvió **HTTP 404**: el modelo configurado en `.env` (`gemini-3-flash`) no existe en el catálogo de Google. El nombre más parecido es `gemini-3-flash-preview`, pero no es el mismo identificador.

El fallo no produjo ningún síntoma durante el Bloque 12 porque **nadie llamaba a ese módulo**. Un valor de configuración incorrecto en un camino de código que nunca se ejecuta no tiene forma de manifestarse.

**Corrección:** se consultó el catálogo real de modelos (`GET /v1beta/models`) y se fijó `gemini-3.8-flash`, un modelo estable —sin el sufijo `preview`— que sí acepta `generateContent`.

**Aprendizaje:** un respaldo que nunca se ha ejercitado no es un respaldo. Es la misma regla que se aplica a los dobles de prueba: la interfaz puede estar bien escrita y, aun así, apuntar a un destino que no existe. Toda integración con un tercero debería correrse al menos una vez contra el proveedor real.

### 9.6. Trazabilidad del checklist de pruebas del cliente

El Bloque 14 pide una suite que cubra cuatro funciones del cliente, con al menos una prueba por cada requisito del checklist. La suite se fue construyendo en los Bloques 12 y 13; esta tabla verifica que no falte ninguno.

| Función | Lo que debe verificar | Prueba que lo cubre |
|---|---|---|
| `enviar_peticion` | Un 401 o 403 lanza `ValueError` | `test_credencial_invalida_lanza_error_autenticacion` (401 y 403) |
| | Un 429 o 500+ lanza `TimeoutError` | `test_falla_momentanea_lanza_error_temporal_con_el_codigo` (429, 500, 502 y 503) |
| | Una respuesta exitosa devuelve el JSON tal cual | `test_respuesta_exitosa_devuelve_el_json` |
| `con_reintento` | Reintenta cuando la función protegida lanza `TimeoutError` | `test_reintenta_y_luego_responde` |
| | No reintenta otro tipo de excepción | `test_con_reintento_no_captura_errores_no_temporales` |
| | Al agotar los intentos, se propaga la excepción original | `test_con_reintento_propaga_el_error_original` |
| `registrar_costo` | Calcula el costo a partir de un conteo de tokens conocido | `test_suma_entrada_y_salida_con_sus_tarifas` |
| | Ese costo se suma al acumulado de la sesión | `test_registrar_costo_acumula_entre_operaciones` |
| `consultar_modelo` | Convierte un `TimeoutError` agotado en `RuntimeError` | `test_agota_los_intentos_y_lanza_error_propio` |
| | Un `ValueError` se propaga sin cambios | `test_pregunta_vacia_lanza_valueerror` y `test_pregunta_que_no_es_texto_lanza_valueerror` |

La suite reúne **133 pruebas**, de las cuales **132 corren por defecto en menos de medio segundo**. La restante, `test_integracion_real_openrouter`, llama al proveedor de verdad y **está excluida por defecto** desde `pytest.ini`, porque consumía cuota en cada corrida y dejaba el resultado a merced de la latencia del servicio. Se ejecuta a propósito con `python -m pytest -m integracion`.

Ese detalle salió del propio consejo del bloque: si la suite tarda varios segundos, alguna prueba está llamando a la red. La nuestra tardaba 4,11 s y la causa era exactamente esa.

**Desviaciones respecto del material, y su motivo.**

1. **Ubicación de los archivos.** El material pide `test_cliente.py` junto a `cliente.py` en la raíz para que `pytest` pueda importar el módulo. Aquí el código vive en `src/` y las pruebas en `tests/`, y `tests/conftest.py` agrega `src/` al `sys.path`. El `ModuleNotFoundError` se evita igual, y separar código de pruebas se vuelve necesario cuando la suite pasa de un archivo.
2. **Las excepciones propias son las del material, con nombre.** El checklist espera `ValueError` para 401/403 y `TimeoutError` para 429/5xx. `ErrorAutenticacion` hereda de `ValueError` y `ErrorTemporal` hereda de `TimeoutError`, así que un `pytest.raises(ValueError)` escrito tal cual el material pasaría; las pruebas fijan la excepción específica, que es más estricta.
3. **El valor exacto de la espera.** El material advierte no verificar cuánto se espera, porque el jitter es aleatorio. Aquí el jitter se neutraliza con una fixture para poder comprobar la progresión del backoff —1, 2 y 4 segundos, y el techo en 6— y aparte se verifica que la parte aleatoria caiga dentro del rango esperado. Sin neutralizarlo, la prueba de la progresión sería imposible; sin la prueba del rango, el jitter quedaría sin cubrir.
4. **`registrar_costo` vive en `src/costos.py`.** Se importa en `cliente.py`, así que `cliente.registrar_costo` existe y la interfaz que el checklist nombra se mantiene. El acumulado se lee con `costos.costo_acumulado()` en lugar de un atributo de módulo, y una fixture `autouse` lo reinicia antes y después de cada prueba para que una prueba no contamine a la siguiente.

### 9.7. Efecto del pipeline sobre respuestas reales

Corrida del **2026-10-31** sobre el proveedor principal: tres consultas, cada una con una referencia escrita a mano. La métrica cuenta la proporción de palabras de la referencia que también aparecen en la respuesta.

| Caso | Texto crudo | Texto procesado |
|---|---:|---:|
| 1. Qué es Python | 85,7 % | **100 %** |
| 2. Qué es una lista | 80,0 % | **88,9 %** |
| 3. Para qué sirve una función | 36,4 % | **36,4 %** |

**El caso 3 es el más honesto de los tres:** el pipeline no cambió nada. La respuesta usó palabras distintas de la referencia —"sirve para agrupar y encapsular instrucciones reutilizables" frente a "recibe parámetros y devuelve un resultado"— y ninguna normalización arregla eso, porque no era un problema de formato sino de contenido.

Los casos 1 y 2 muestran lo contrario: el modelo dijo lo mismo con otra puntuación, y la limpieza recuperó entre 8 y 14 puntos de coincidencia.

**Dos límites de la métrica, verificados en la misma corrida sin llamar a la API:**

| Límite | Comparación | Resultado |
|---|---|---|
| Orden de las palabras | "El perro persiguió al gato" frente a "El gato persiguió al perro" | **100 %**, con significados opuestos |
| Repeticiones | "el gato" frente a "el gato el gato el gato" | **100 %**: se compara por conjuntos |

Ninguno de los dos es un error de implementación: son propiedades del tipo de métrica. El primero se resuelve con criterio humano al leer la respuesta; el segundo, contando repeticiones con `Counter` en lugar de `set`.

El registro completo, con las respuestas textuales, está en `docs/registro_pipeline_texto.txt`. Costo de la corrida: **US$ 0,00019**.

### 9.8. Precisión, cobertura y F1 sobre respuestas reales

Corrida del **2026-10-31** sobre cinco consultas, cada una con su referencia escrita a mano. Las tres métricas se calculan sobre el texto ya normalizado por el pipeline del Bloque 15.

| # | Consulta | Precisión | Cobertura | F1 |
|---|---|---:|---:|---:|
| 1 | Qué es Python | 0,27 | 1,00 | 0,42 |
| 2 | Cuándo se creó Python | 0,24 | 1,00 | 0,38 |
| 3 | Qué es una lista | 0,38 | 0,90 | 0,53 |
| 4 | Para qué sirve una función | 0,14 | 0,25 | **0,18** |
| 5 | Qué es una variable | 0,39 | 0,82 | 0,53 |
| | **Promedio** | **0,28** | **0,79** | **0,41** |

**La lectura importante está en la diferencia entre precisión y cobertura.** Cobertura 0,79 contra precisión 0,28: el modelo dice casi todo lo que la referencia esperaba, pero con muchas palabras de más. Pese a la instrucción "responde en una sola oración", las respuestas vinieron largas.

Eso cambia qué métrica mirar: si la referencia es corta y específica, la precisión es la que manda, y aquí es la que está mal.

**Los dos casos que valen la pena leer:**

- **Caso 2 (F1 0,38).** La respuesta fue "Python fue creado por Guido van Rossum a finales de los años 80 y su primera versión se publicó en 1991". Es **correcta**, y su cobertura es 1,00. La precisión la castiga por extensa. Un F1 bajo aquí no significa un error: significa que la respuesta dice más de lo que la referencia pedía.
- **Caso 4 (F1 0,18).** La respuesta fue "una función sirve para encapsular una tarea o cálculo reutilizable a partir de unas entradas". Habla de lo mismo que la referencia, con otras palabras. Es el límite de los sinónimos, apareciendo en una respuesta que nadie diseñó para ilustrarlo.

**Los tres límites de la métrica, fijados como pruebas:**

| Límite | Comparación | F1 |
|---|---|---:|
| Sinónimos | "El auto es veloz" frente a "El vehículo es rápido" | 0,50 |
| Dato equivocado | "creado en 1991" frente a "creado en 1989" | 0,80 |
| Negación ausente | "Python no es un lenguaje compilado" frente a "Python es un lenguaje compilado" | 0,91 |

Están en `tests/test_metricas.py` a propósito: si alguien "mejora" la métrica para que el sinónimo dé 1,0, las pruebas fallan y explican por qué no corresponde. La métrica mide **parecido léxico**, no corrección; sirve para decidir qué leer primero, no para aprobar una respuesta sin mirarla.

El detalle completo, con las respuestas textuales, está en `docs/resultados_metricas.json`. Costo de la corrida: **US$ 0,00033**.

> **Nota:** el Bloque 17 extendió la misma corrida a **20 casos** sobre el mismo archivo, para poder elegir una muestra de revisión manual. El resumen y los hallazgos de esa ampliación están en la sección 9.9.

### 9.9. Revisión cualitativa de la muestra (Bloque 17)

La corrida se extendió a **20 consultas** y se revisaron **10 a mano**: los 5 casos con F1 más bajo y los 5 con F1 más alto, excluyendo los del medio.

**Promedio de las 20 respuestas:** F1 **0,55** · precisión **0,43** · cobertura **0,83**. La brecha entre precisión y cobertura se mantiene: el cliente dice casi todo lo que se espera, con muchas palabras de más.

| Consulta | F1 | Corrección factual | Completitud | Contradicción |
|---|---:|---|---|---|
| Sigla PEP | 0,13 | **no** | **no** | no |
| Para qué sirve una función | 0,30 | sí | parcialmente | no |
| Qué es un bucle for | 0,33 | sí | sí | no |
| Cómo se abre un archivo | 0,35 | sí | sí | no |
| Qué es la indentación | 0,36 | sí | sí | no |
| Qué es PEP 8 | 0,73 | sí | sí | no |
| Qué es un comentario | 0,78 | sí | sí | no |
| Qué es Python | 0,79 | sí | sí | no |
| Quién creó Python | 1,00 | sí | sí | no |
| Año de la primera versión | 1,00 | sí | sí | no |

**El resultado que importa: 3 de los 5 casos con F1 más bajo eran respuestas correctas y completas.** La métrica no estaba señalando un problema del cliente; estaba señalando que la respuesta era más larga que la referencia.

**Y el caso más bajo de todos engaña en el otro sentido.** La sigla PEP se respondió con el significado del ámbito financiero ("Persona Expuesta Políticamente"), no con "Python Enhancement Proposal". El F1 de 0,13 coincidió con el error, pero por casualidad: reaccionó a que no compartía palabras, no a que el contenido fuera incorrecto.

**Dos modos de falla que no estaban en el material y aparecieron acá:**

1. **El código diluye la precisión.** La respuesta sobre abrir archivos incluía `with open(...)`. Es correcta y más completa que la referencia, pero sus tokens de código no existen en la referencia y bajaron el F1 a 0,35.
2. **Un F1 perfecto mide acuerdo con la referencia, no verdad.** La respuesta `1991` contra la referencia `1991` da 1,00. Si la referencia estuviera mal escrita, una respuesta errónea habría sacado lo mismo. La métrica compara dos textos humanos; no tiene acceso a la verdad.

Todo quedó documentado en `docs/limites_metrica.md`, con la rúbrica aplicada caso por caso en `revision_manual.py` y `docs/revision_manual.json`.

### 9.10. Verificación del índice vectorial (Bloque 24)

El corpus son **6 documentos** del proyecto, partidos en **31 fragmentos** con `chunk_size=400` y `chunk_overlap=60`. La partición se decidió con una medición previa: el splitter corta en párrafos, así que bajar el tamaño a 200 duplica los fragmentos (85) sin ganar precisión, y subirlo a 600 mezcla temas distintos dentro del mismo fragmento.

La prueba de que el índice **persistió** no es que el script de indexación termine sin error, sino que un proceso distinto se reconecte al mismo directorio y cuente lo mismo:

```bash
python experimentos/verificar_indice.py
```

| Comprobación | Resultado |
|---|---|
| Fragmentos en disco (proceso nuevo) | 31 |
| Modelo registrado en la colección | `models/gemini-embedding-001` |
| Llamadas de embedding al indexar | 1 (un solo lote) |

**Búsqueda por significado**, con preguntas que no comparten palabras con el documento que deberían encontrar:

| Pregunta | Primer resultado | Distancia |
|---|---|---:|
| ¿Qué pasa si el proveedor se cae y la llamada falla varias veces? | `01_cliente_modular.md` → *Backoff exponencial* | 0,661 |
| ¿Cuánto cuesta cada llamada al modelo? | `02_costos.md` → *Costo por operación* | 0,572 |
| ¿Cómo se comprueba que el código funciona antes de publicarlo? | `05_pruebas.md` → *Trazabilidad* | 0,590 |

La primera consulta **no** trae como primer resultado el fragmento de backoff, sino uno de pruebas: es un caso real de recuperación imperfecta y queda anotado para el conjunto de evaluación del Bloque 27, donde se mide en lugar de mirarse a ojo.

### 9.11. Verificación del flujo de recuperación con citas (Bloque 26)

Dos corridas reales contra el índice de 31 fragmentos, con generación en DeepSeek:

**Pregunta respondible** — *¿Qué pasa si la llamada al proveedor falla varias veces?*

| Aspecto | Resultado |
|---|---|
| Fragmentos recuperados | 3 (uno de pruebas, dos de cliente modular) |
| Citas en la respuesta | `[3]` para los reintentos y la espera creciente, `[2]` para el backoff |
| Citas contra el mapa | las dos existen y apuntan al archivo correcto |
| Límite admitido | agregó que el contexto no dice cuántos reintentos hay ni qué pasa al agotarlos |

**Pregunta imposible** — *¿Cuál es el precio del dólar hoy en Chile?*

La respuesta fue: el contexto no incluye esa información y no puede responder con las fuentes dadas, citando los tres fragmentos que revisó. No inventó una cifra. Es la verificación que importa: un sistema que siempre responde algo no distingue entre saber y no saber.

**Trazabilidad de una cita, verificada a mano:** `[3]` apuntaba a `01_cliente_modular.md#0`, y ese fragmento efectivamente describe `con_reintento()` como función interna. La cita es correcta, no decorativa.

### 9.12. Evaluación de la recuperación: precisión y recall (Bloque 27)

15 preguntas sobre el corpus: 13 con un fragmento que las responde y **2 imposibles** (licencia del proyecto y número de alumnos), para comprobar que el sistema no fuerza una respuesta donde no la hay. Las etiquetas se decidieron leyendo los 31 fragmentos, antes de ver qué devolvía el retriever.

| Configuración | Precisión@k | Recall@k | Archivo |
|---|---:|---:|---|
| `k=3` (referencia) | **0,289** | **1,000** | `resultados_evaluacion.txt` |
| `k=5` | 0,173 | 1,000 | `resultados_evaluacion_k5.txt` |

**El recall 1,000 es el resultado que importa:** en las 13 preguntas respondibles, el fragmento correcto apareció siempre entre los recuperados. La recuperación no falla en ninguna de las 13.

**La precisión de 0,289 no es un defecto y leerla como tal sería el error.** Con `k=3` y un solo fragmento relevante por pregunta, la precisión máxima alcanzable es 1/3 = 0,33. La cuenta lo confirma: 13 casos a 0,33 y 2 a 0,00 dan exactamente 0,289. Es decir, **cada pregunta respondible obtuvo la precisión máxima posible**; el 0,29 solo refleja los dos lugares que sobran en el top-3.

**Con `k=5` la precisión baja a 0,173, que es 1/5 = 0,20 evaluado sobre 13 casos.** El recall no mejora porque ya estaba en 1,000. Conclusión honesta: en este corpus, subir `k` no mejora nada y solo trae fragmentos irrelevantes al contexto —más tokens de entrada y más ruido para el modelo—. El valor óptimo es `k=3`, y ahora está medido, no supuesto.

**Límite del método, declarado:** precisión@k con un solo fragmento relevante tiene techo 1/k, así que comparar precisión entre valores distintos de `k` mide sobre todo cuántos lugares sobra, no calidad de búsqueda. Dos corridas seguidas con la misma configuración dieron archivos idénticos, así que la recuperación es determinista y las diferencias entre configuraciones son reales.

## 10. Registro del desarrollo y ensayo de instalación limpia

- **Fecha:** 2026-10-23
- **Desarrollo:** creación del banco de prompts, elaboración de un notebook y preparación de pruebas automatizadas.
- **Pruebas:** implementación de pruebas unitarias, archivo `conftest.py` para la configuración compartida de las pruebas y pruebas de consistencia de los prompts.
- **Hallazgo:** después de publicar el Bloque 10, descubrí que faltaba incluir `pytest` en `requirements.txt`, porque había incorporado las pruebas posteriormente.
- **Corrección:** actualicé `requirements.txt` y registré el cambio en Git antes de iniciar el Bloque 11.

##### Bloque 11: Ensayo de instalación limpia

- **Procedimiento:** cloné mi repositorio, instalé las dependencias y configuré las variables de entorno mediante el archivo `.env`.
- **Resultado:** ejecuté `main.py` sin errores en el repositorio clonado.
- **Validación de consistencia:** ejecuté `prompts/test_consistencia.py` y obtuve 9 de 9 respuestas válidas (100 % de éxito).
- **Evidencia:** notebook del proyecto, pruebas automatizadas, `conftest.py` y archivo `prompts/registro_resultados.json` generado durante las pruebas.

##### Bloque 12: Cliente modular con clasificación de errores y reintento

- **Ubicación en el programa:** primer bloque del Gate 2, dentro del Módulo 2 (Operatividad de aplicaciones generativas: APIs, orquestación, RAG y personalización).
- **Objetivo:** agrupar el consumo del modelo detrás de una única función pública que clasifique los errores del proveedor y reintente las fallas temporales, sin repetir esa lógica en cada parte del proyecto que consulte la API.
- **Implementación:** creé `src/errores.py` con `ErrorAutenticacion`, `ErrorTemporal` y `ErrorReintentosAgotados`; `src/cliente.py` con `consultar_modelo()` como entrada pública y `con_reintento()` como lógica interna; y llevé `src/consumir_deepseek.py` al mismo estándar, con timeout, instrucción de sistema y clasificación de respuestas. Agregué `src/consumir_gemini.py` como proveedor de respaldo con el mismo contrato de llamada.
- **Decisión:** las excepciones propias heredan de `ValueError`, `TimeoutError` y `RuntimeError` para que los manejos de error que ya existían en el proyecto sigan funcionando sin modificaciones.
- **Diferencia con el material del bloque:** el ejercicio se plantea sobre la API de Gemini. Lo implementé sobre DeepSeek, que es el proveedor definido en `docs/decision-arquitectura.md`, y dejé Gemini como respaldo. El patrón solicitado —función pública única, clasificación de errores, reintento interno y error propio al agotar los intentos— se mantiene igual.
- **Hallazgo (clasificación):** con una clave inválida, DeepSeek responde 401, pero Gemini responde 400 con `error.details[].reason = "API_KEY_INVALID"`. Si hubiera clasificado solo por el código HTTP, el error de credenciales de Gemini habría quedado registrado como un error genérico.
- **Hallazgo (intérprete):** al ejecutar una prueba manual con el lanzador `py`, el intérprete utilizado fue el del sistema operativo y no el del entorno virtual, por lo que faltaban dependencias. Para resolverlo instalé las librerías y regeneré `requirements.txt` con `pip freeze`, lo que sobrescribió el archivo con las dependencias del Python global: se perdieron las versiones fijadas, desapareció `pytest` y se agregó `dotenv==0.9.9`, un paquete distinto de `python-dotenv` que ocupa el mismo nombre de módulo.
- **Corrección:** restauré `requirements.txt` desde el historial de Git después de comprobar que su contenido coincidía con las dependencias del entorno virtual. El procedimiento correcto quedó documentado en la sección 6.1. El hallazgo se registró como error recurrente, porque ya había ocurrido antes con otro intérprete.
- **Corrección adicional:** la validación de entrada comprobaba únicamente que la pregunta no estuviera vacía. Con un valor que no fuera texto, el programa fallaba con `AttributeError` en lugar de informar el problema. Agregué una comprobación de tipo con un mensaje explícito.
- **Evidencia:** 115 pruebas automatizadas que no realizan llamadas reales, más las verificaciones manuales de la sección 9.4.

##### Bloque 13: Backoff exponencial, jitter y control de costo por operación

- **Objetivo:** reemplazar la espera fija entre reintentos por un backoff exponencial con techo y aleatoriedad (jitter), e incorporar el registro de costo acumulado por operación a partir de los tokens informados por el proveedor.
- **Implementación:** creé `src/costos.py` con la tabla oficial de precios por componente (cache hit, cache miss, salida) y el selector de hora punta y hora valle en UTC. Actualicé `con_reintento` en `src/cliente.py` con una progresión geométrica de la espera, techo de 30s y 10% de jitter. Amplié la suite con 18 tests nuevos en `tests/test_costos.py` y el ajuste de `tests/test_cliente.py` (133 pruebas en total).
- **Decisión (tarifas):** el material del bloque propone usar un único costo promedio por 1.000 tokens. DeepSeek, en cambio, cobra distinto la entrada con caché, la entrada sin caché y la salida, además de duplicar su tarifa en horas punta. Usar un número único habría generado un error de estimación de hasta 13x. Se optó por un cálculo desglosado fiel a la documentación oficial.
- **Decisión (rate limit):** el ejercicio sugiere provocar un HTTP 429 de forma intencional mediante un bucle rápido. Sin embargo, la documentación de DeepSeek especifica que no aplica un límite de peticiones por minuto (RPM), sino un límite estricto de concurrencia de 2.500 peticiones simultáneas. Por ello, el comportamiento del 429 y del backoff se validó de forma rigurosa mediante dobles de red y pruebas automatizadas, documentando el desvío operativo respecto al proveedor.
- **Ensayo de rate limit:** se ejecutó contra Gemini, el proveedor de respaldo, porque DeepSeek no limita por minuto. El primer intento falló con un error temporal y el backoff esperó **1,00 s → 2,10 s → 4,07 s**: la progresión 1-2-4 con el jitter visible en los decimales. En una segunda corrida se confirmó el techo: **1,10 s → 2,04 s → 4,01 s → 6,37 s**, es decir, dejó de duplicar al llegar al máximo de 6 s en vez de seguir a 8 y 16. Ambas corridas se interrumpieron, una por consumo de cuota y otra por timeouts encadenados.
- **Hallazgo (diagnóstico):** el mensaje de error agrupaba el 429 y los 5xx en una misma frase sin el código HTTP, así que no permitía distinguir un límite de uso de una caída del servidor. Se corrigió para que el mensaje incluya el código recibido, en los dos proveedores. Sin ese dato, el ensayo no puede afirmar qué falló: solo que fue un error temporal.
- **Evidencia:** 133 pruebas unitarias y de comportamiento en verde, ejecutadas en entorno aislado sin alterar la interfaz pública del cliente.

##### Bloque 14: Pruebas unitarias del cliente modular

- **Objetivo:** una suite que cubra el checklist del bloque, función por función, sin tocar la red y sin esperar de verdad.
- **Implementación:** la suite ya existía de los bloques 12 y 13; el trabajo de este bloque fue verificarla contra el checklist y completar lo que faltaba. Se agregó una prueba de contrato que fija las cuatro funciones públicas, y `pytest.ini` para excluir por defecto el único test que llama al proveedor real.
- **Hallazgo (cuota):** `test_integracion_real_openrouter` llamaba a OpenRouter en **cada** corrida de la suite y era la causa de que tardara 4,11 s. Quedó marcado como `integracion` y excluido por defecto; se ejecuta a propósito con `python -m pytest -m integracion`. La suite pasó a 0,47 s.
- **Diferencia con el material:** los archivos viven en `src/` y `tests/` en lugar de la raíz, y `tests/conftest.py` agrega `src/` al `sys.path` para evitar el `ModuleNotFoundError`. Las excepciones propias heredan de `ValueError` y `TimeoutError`, así que las pruebas escritas tal como las plantea el material pasarían igual.
- **Evidencia:** 133 pruebas en verde, trazabilidad del checklist en la sección 9.6.

##### Bloque 15: Pipeline de limpieza y tokenización de texto

- **Objetivo:** dejar el texto en un formato comparable antes de medir nada, y comprobarlo sobre respuestas reales del cliente.
- **Implementación:** `src/pipeline_texto.py` con `limpiar_texto`, `tokenizar`, `procesar_texto` y `vectorizar`; `experimentos/medir_pipeline.py` para la corrida con respuestas reales; registro en `docs/registro_pipeline_texto.txt`.
- **Decisión (limpieza):** la puntuación se **elimina**, no se reemplaza por espacio, así que `"hola,mundo"` queda `"holamundo"`. Se mantiene porque el material usa esa expresión y porque la comparación es por palabras, pero se documenta y se prueba: es una consecuencia del diseño, no un descuido.
- **Hallazgo:** sobre tres respuestas reales, el pipeline recuperó entre 8 y 14 puntos de coincidencia en dos casos y **ninguno en el tercero**, donde la diferencia era de contenido y no de formato. Ese caso es la mejor evidencia de que la limpieza no arregla el significado.
- **Límites verificados:** el orden de las palabras y las repeticiones dan 100 % de coincidencia con significados opuestos o distintos. El primero se resuelve leyendo la respuesta; el segundo, con conteo en lugar de conjuntos.
- **Evidencia:** 21 pruebas nuevas (154 en total), registro de la corrida y costo de US$ 0,00019.

##### Bloque 16: Métricas de precisión, cobertura y F1

- **Objetivo:** pasar de "coincide o no" a "cuánto coincide", contando repeticiones en lugar de comparar conjuntos.
- **Implementación:** `src/metricas.py` con `contar_coincidencias` (intersección de `Counter`, que toma el mínimo de cada palabra), `calcular_precision_cobertura_f1`, `calcular_metricas` (aplica el pipeline del Bloque 15 de punta a punta), `resumir_metricas` y `ordenar_por_f1`. `experimentos/medir_metricas.py` corre cinco consultas reales y guarda `docs/resultados_metricas.json`.
- **Decisión (validación de entrada):** `contar_coincidencias` rechaza un `str` con `TypeError`. Sin eso, `Counter` cuenta cada carácter y devuelve un número equivocado **sin fallar**, que es la peor combinación posible.
- **Hallazgo:** el promedio de las cinco respuestas fue **precisión 0,28 y cobertura 0,79**. La brecha indica respuestas completas pero extensas: la instrucción "responde en una sola oración" no se cumplió y la precisión es la que lo delata.
- **Hallazgo (límites en producción):** un caso correcto quedó con F1 0,38 por ser más largo que la referencia, y otro con F1 0,18 por usar sinónimos. Son los límites del bloque, apareciendo en respuestas que nadie diseñó para ilustrarlos.
- **Decisión (límites como pruebas):** los tres casos del material —sinónimo, dato equivocado y negación ausente— quedaron como tests. No verifican que la métrica funcione, sino que **siga midiendo lo mismo**: si alguien intenta "arreglarla" para que el sinónimo dé 1,0, la suite lo detiene y explica por qué.
- **Evidencia:** 18 pruebas nuevas (172 en total), resultados guardados con las respuestas incluidas, costo de US$ 0,00033.

##### Bloque 17: Interpretación cualitativa y límites de la métrica

- **Objetivo:** leer una muestra de las respuestas con una rúbrica fija y dejar por escrito qué no puede medir el pipeline, para que quien lo herede no confíe en el número donde no corresponde.
- **Implementación:** la corrida se extendió a **20 consultas**; se revisaron **10 a mano** —los 5 con F1 más bajo y los 5 con F1 más alto, sin los del medio— con la rúbrica de corrección factual, completitud y contradicción. El registro vive en `revision_manual.py` y `docs/revision_manual.json`; el documento de límites, en `docs/limites_metrica.md`.
- **Hallazgo:** **3 de los 5 casos con F1 más bajo eran respuestas correctas y completas.** El número estaba midiendo extensión, no calidad. Con referencias cortas como las de este proyecto, la precisión dice más que el F1: el promedio fue 0,43 de precisión contra 0,83 de cobertura.
- **Hallazgo (falso negativo):** el único caso realmente incorrecto —la sigla PEP respondida con el significado financiero— tuvo el F1 más bajo de la corrida, 0,13, pero por casualidad: la métrica reaccionó a la falta de palabras en común, no al error de contenido.
- **Hallazgo nuevo (modo de falla no documentado en el material):** un F1 de 1,00 mide acuerdo con la referencia que escribió una persona, no verdad. Si la referencia está mal, la métrica premia la respuesta equivocada. Se agregó como modo de falla en `docs/limites_metrica.md`.
- **Hallazgo nuevo (dilución por código):** las respuestas con ejemplos de código meten tokens que no existen en la referencia y bajan la precisión sin que haya error.
- **Decisión:** la rúbrica se registra en un `.py` y se exporta a JSON con `json.dump`, para que el registro sea legible por una persona y consumible por un programa, sin duplicar la fuente.
- **Evidencia:** 172 pruebas en verde, 20 respuestas medidas, 10 revisadas a mano, costo acumulado de la corrida US$ 0,00150.

##### Mejora posterior al Gate 1: excepción propia para el límite de uso (2026-11-02)

- **Origen:** la devolución del Gate 1 pidió "aislar excepciones específicas de red (como `RateLimitError`)". Hasta este cambio, un 429 y un 5xx se levantaban como el mismo `ErrorTemporal`.
- **Implementación:** `ErrorLimiteDeUso` hereda de `ErrorTemporal`, así que todo `except ErrorTemporal` que ya existía sigue funcionando sin cambios. Los dos proveedores levantan esa subclase **solo** en 429; los 5xx siguen siendo `ErrorTemporal`.
- **Por qué importa:** un 429 pide bajar el ritmo; un 5xx no se arregla esperando distinto. Sin la subclase, quien consume el cliente tenía que leer el texto del mensaje para distinguirlos.
- **Cobertura:** cuatro pruebas por proveedor: la subclase en 429, los 5xx sin confundirse con el límite, y la compatibilidad con `except ErrorTemporal` y `except TimeoutError`.
- **Deuda pendiente:** respetar el encabezado `Retry-After` cuando el proveedor lo envíe, en lugar de la espera calculada por el backoff. Queda anotado en la sección 11.

##### Bloque 24: Indexación del corpus con chunking, embeddings y persistencia en Chroma

- **Objetivo:** juntar las dos piezas del bloque —partir texto en fragmentos y calcular embeddings reales— sobre un corpus propio, y dejar el índice persistido en disco para que la recuperación lo consulte sin recalcular nada.
- **Implementación:** `corpus/` con **seis documentos** del proyecto (cliente modular, costos, banco de prompts, agente, pruebas y búsqueda/pipeline). `indexar_corpus.py` con `cargar_documentos`, `partir_documentos`, `construir_ids`, `verificar_fragmentos`, `indexar_corpus`, `abrir_indice` y `verificar_indice`. Persistencia en `vectorstore/` con Chroma y embeddings de `models/gemini-embedding-001`.
- **Decisión (proveedor de embeddings):** DeepSeek no ofrece endpoint de embeddings, así que Gemini deja de ser solo el respaldo de generación y pasa a ser el proveedor de embeddings del proyecto. El flujo RAG queda con dos dependencias de proveedor a la vez, y eso está declarado en la sección 11.
- **Decisión (parámetros de partición):** 400 caracteres con 60 de solapamiento, elegidos midiendo. Con 200 el corpus daba 85 fragmentos —el splitter corta en párrafos, así que el límite real lo ponen los saltos de línea— y con 600 algunos fragmentos mezclaban dos temas. Con 400/60 quedaron 34 fragmentos, y tras fusionar las colas huérfanas quedaron 31, casi todos de un párrafo completo, y el solapamiento solo actúa cuando un párrafo se parte.
- **Decisión (metadata e ids desde el primer indexado):** el material agrega `metadata` e `ids` en el Bloque 26 y obliga a reindexar. Se hicieron ahora, por dos razones: `fuente` no se puede agregar retroactivamente sin volver a pagar los embeddings, y los ids predecibles (`archivo.md#3`) son lo que permite etiquetar a mano el conjunto de evaluación del Bloque 27 sin depender del orden interno de Chroma.
- **Decisión (modelo declarado en la colección):** el índice guarda en su metadata el modelo de embeddings, el tamaño de chunk y el solapamiento. Un índice vectorial no dice con qué modelo se construyó: cambiar de modelo sin reindexar no falla, y si la dimensión coincide devuelve fragmentos sin relación con la pregunta mientras el sistema parece funcionar. `verificar_indice` compara el modelo guardado con el configurado y corta si no coincide.
- **Hallazgo (log engañoso):** la primera corrida imprimió `Indexando 34 fragmentos con modelo sin declarar` aunque sí lo declaraba, porque el mensaje leía el parámetro opcional antes de aplicar el valor por defecto. Corregido antes del commit; el dato importa porque es el que se mira para saber con qué se construyó el índice.
- **Hallazgo (verificación de integridad):** los fragmentos se revisan **antes** de llamar a la API —vacíos, sin `fuente` y con texto repetido—, porque descubrir un fragmento vacío después de indexar obliga a reindexar y a pagar de nuevo. Un fragmento duplicado no lanza ningún error: ocupa dos lugares entre los candidatos y desplaza contenido que sí aporta.
- **Evidencia:** 21 pruebas nuevas con embeddings falsos, incluida la persistencia real en un directorio temporal y la reconexión sin volver a embeber (288 en total). Corrida real: 6 documentos, 31 fragmentos, **una sola llamada de embedding** en lote, y reconexión desde otro proceso con el conteo y la metadata correctos (sección 9.10).

##### Bloque 26: Recuperación semántica y prompt dinámico con citas

- **Objetivo:** pasar de la búsqueda por palabra clave de `src/herramientas.py` a una búsqueda por significado, y armar con los fragmentos un prompt que obligue al modelo a citar de dónde sale cada afirmación.
- **Implementación:** `recuperacion.py` con `formatear_contexto`, `recuperar`, `construir_prompt`, `responder`, `citas_usadas` y `verificar_citas`. Se reconecta al índice con `abrir_indice` en vez de reindexar, y reutiliza `verificar_indice` para cortar si el índice se construyó con otro modelo de embeddings.
- **Decisión (k=3):** con 31 fragmentos de un párrafo, tres cubren la pregunta y su contexto inmediato. Más fragmentos diluyen la atención del modelo sobre los que importan; menos pierden el contexto que la cita necesita. El valor queda como constante configurable y el Bloque 27 mide el trade-off en lugar de confiarlo a la intuición.
- **Decisión (las dos instrucciones juntas):** el prompt pide citar y, en la misma frase, admitir cuando el contexto no alcanza. "Cita tus fuentes" sin "admite cuando no sabes" produce citas inventadas para llenar el hueco.
- **Decisión (verificar citas sin bloquear):** `verificar_citas` compara los números citados con el mapa real y devuelve problemas, pero no corta la respuesta. Bloquear obligaría a reintentar una llamada ya pagada; informar deja el problema visible en la traza, que es donde se revisa.
- **Hallazgo (verificación en dos direcciones):** además de comprobar que la respuesta correcta cita bien, se probó con una pregunta **imposible** («¿cuál es el precio del dólar hoy en Chile?»). El modelo respondió que el contexto no alcanza y citó los tres fragmentos que revisó, sin inventar una cifra. Sin esa segunda prueba, un sistema que siempre responde algo parecería funcionar.
- **Hallazgo (el patrón de cita se rompe con un recorrido manual):** la primera versión extraía las citas recorriendo carácter por carácter y acumulaba dígitos fuera de corchetes, así que un año suelto seguido de `]` se contaba como cita. Se reemplazó por un patrón regular, y quedó una prueba con `1991` entre paréntesis para que no vuelva.
- **Diferencia con el material:** el material reindexa en este bloque para agregar `metadata` e `ids`. Como se agregaron en el Bloque 24, aquí no hubo reindexación ni un segundo pago de embeddings; solo se verificó que el índice existente sirviera.
- **Evidencia:** 22 pruebas nuevas (310 en total) y las dos corridas reales de la sección 9.11, con el mapa de citas y la verificación manual de una cita.

##### Bloque 27: Evaluación de la recuperación con precisión y recall

- **Objetivo:** dejar de confiar en "a simple vista parece que trae lo correcto" y medir la recuperación con un conjunto de preguntas etiquetadas a mano.
- **Implementación:** `evaluacion_recuperacion.py` con `precision_at_k`, `recall_at_k`, `promediar`, `evaluar_recuperacion`, `formatear_resultados` y `escribir_resultados`; 15 casos en `CASOS_EVALUACION` (13 respondibles y 2 imposibles); corridas guardadas en `resultados_evaluacion.txt` y `resultados_evaluacion_k5.txt`. `experimentos/listar_fragmentos.py` imprime los ids con su texto, que es lo que hace posible etiquetar leyendo el corpus.
- **Hallazgo previo (fragmentos huérfanos):** al listar los fragmentos para etiquetar aparecieron colas de una línea (`"incorrecto es peor que ningún material."`, 30 caracteres) y un fragmento que era solo el título del documento. No responden nunca, pero ocupan un lugar entre los candidatos y desplazan a uno que sí. Se agregó una fusión de fragmentos cortos con el vecino del mismo documento, con tope para no mezclar temas: 34 → 31 fragmentos, mínimo 174 caracteres. Se corrigió **antes** de etiquetar, porque etiquetar sobre fragmentos basura es etiquetar basura.
- **Decisión (un fragmento relevante por pregunta):** 13 de los 15 casos tienen un solo id relevante. No es pereza: con un id, el recall no puede ser menor que la precisión, y esa relación funciona como control de que las fórmulas y el `k` están bien usados. Dos casos quedan con el conjunto vacío —preguntas que el corpus no responde— y su recall sale `n/a`, excluido del promedio en lugar de contado como 0.
- **Decisión (no sobrescribir):** `escribir_resultados` lanza `FileExistsError` si el archivo existe. Sobrescribir deja la corrida nueva y borra la referencia contra la que se quería comparar, que es exactamente lo que le da sentido a la medición.
- **Decisión (leer la configuración del índice):** el material pide anotar a mano `k`, `chunk_size` y `chunk_overlap` en el archivo. Se leen de la metadata de la colección y se agrega el modelo de embeddings. Anotarlos a mano es un error silencioso esperando a ocurrir: dos corridas comparadas con la configuración mal escrita.
- **Resultado (k=3):** precisión **0,289**, recall **1,000**. El fragmento correcto apareció en el top-3 en las **13** preguntas respondibles.
- **Hallazgo (la precisión tiene techo):** con `k=3` y un fragmento relevante, el máximo de precisión es 1/3 = 0,33. Los 13 casos respondibles dieron exactamente 0,33 y los 2 imposibles 0,00, que da 0,289. Es decir, la precisión fue **máxima en todos los casos**, y el número bajo solo cuenta los dos lugares que sobran.
- **Resultado (k=5):** precisión 0,173, recall 1,000. La precisión baja porque su techo pasa a 1/5, y el recall no mejora porque ya estaba en 1. Conclusión medida: subir `k` en este corpus no aporta y mete ruido al prompt. `k=3` queda justificado con números, no con intuición.
- **Verificación (determinismo):** dos corridas consecutivas con la misma configuración produjeron archivos idénticos, así que las diferencias entre configuraciones son reales y no ruido del servicio.
- **Evidencia:** 20 pruebas nuevas (335 en total) y las tres corridas reales de la sección 9.12.

## 11. Limitaciones y mejoras futuras
Entre las posibles mejoras se encuentran:
- Ampliar el conjunto de comentarios de prueba.
- Incorporar más casos ambiguos y situaciones límite.
- Profundizar en la validación semántica de las respuestas.
- Mejorar el registro y análisis de errores.
- Evaluar el comportamiento de diferentes modelos.
- Medir los costos asociados al consumo de la API.
- Evitar la duplicación del banco de prompts entre los distintos componentes del proyecto.
- Permitir que el cliente alterne entre proveedores sin modificar el código que lo consume.
- Respetar el encabezado `Retry-After` del proveedor cuando responde 429, en lugar de la espera calculada por el backoff.
- Modelar los feriados del proveedor al calcular la tarifa de hora punta.
- Registrar el costo en un archivo o base de datos para conservarlo entre ejecuciones.
- **Dependencia de dos proveedores en el flujo RAG:** la generación usa DeepSeek y los embeddings usan Gemini, porque DeepSeek no ofrece embeddings. Si Gemini no responde, la recuperación no funciona aunque la generación sí: es una dependencia que hay que declarar y, más adelante, decidir si se mitiga con embeddings locales.
- **El índice no se versiona y hay que reindexar cuando cambia el corpus:** `vectorstore/` está en `.gitignore` por ser contenido reconstructible. La contrapartida es que un clon nuevo del repositorio no trae índice y, si se edita `corpus/`, el índice queda desactualizado hasta correr `python indexar_corpus.py`.
- **El corpus es una foto de la documentación, no la documentación:** los seis documentos de `corpus/` están escritos a mano y hay que mantenerlos sincronizados con el código. Un cambio en `src/costos.py` que no se refleje en `corpus/02_costos.md` produce respuestas correctas sobre un sistema que ya no existe.
- **La precisión@k tiene techo 1/k:** con un solo fragmento relevante por pregunta, comparar precisión entre valores distintos de `k` mide cuántos lugares sobran, no la calidad de la búsqueda. Una comparación honesta entre configuraciones exige mirar el recall, o usar una métrica que no dependa de `k` (por ejemplo MRR o nDCG).
- **El conjunto de evaluación tiene 15 casos y todos de un solo fragmento:** mide bien el caso "una pregunta, un fragmento que responde", y no cubre preguntas cuya respuesta está repartida en dos documentos. Ampliarlo a casos multi-fragmento exigiría otra relación de control, porque con varios ids relevantes el recall ya puede superar a la precisión.

## 12. Conclusión

El proyecto integra un banco de tres prompts especializados con una API de inteligencia artificial generativa para analizar comentarios de encuestas de satisfacción.

La ejecución principal permite comprobar el funcionamiento del flujo de consultas, mientras que el script de consistencia amplía las pruebas mediante múltiples intentos, validación estructural y registro de resultados.

Las ejecuciones realizadas alcanzaron un 100 % de éxito en ambos procesos. Estos resultados permiten verificar el funcionamiento de las validaciones implementadas en los casos evaluados, sin garantizar la corrección semántica de todas las respuestas futuras.

El proyecto constituye una base para continuar desarrollando y evaluando soluciones de análisis de encuestas asistidas por inteligencia artificial.