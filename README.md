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
plantilla-repositorio-base/
├── docs/
│   ├── decision-arquitectura.md
│   ├── Diagrama de flujo para consumir OpenRouter.png
│   ├── Evidencia_peticion_autenticada_Thunder_Client.png
│   └── Evidencia_peticion_evidencia_autenticada_postman.png
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
│   └── parsear_json.py
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
│   └── test_parsear_json.py
├── experimentos/
│   └── ensayo_rate_limit.py
├── .env.example
├── .gitignore
├── main.py
├── pytest.ini
├── README.md
└── requirements.txt

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
| `src/errores.py` | Define las excepciones propias del cliente y su relación con las excepciones estándar de Python. |
| `src/costos.py` | Calcula el costo de cada operación a partir de los tokens informados por el proveedor, con tarifas por componente y distinción entre hora punta y hora valle, y mantiene el acumulado de la sesión. |
| `src/parsear_json.py` | Procesa las respuestas JSON y comprueba las claves esperadas. |
| `tests/conftest.py` | Agrega `src/` al `sys.path` y reinicia el acumulado de costo antes y después de cada prueba. |
| `pytest.ini` | Excluye por defecto el test que llama al proveedor real, para que la suite no consuma cuota. |
| `experimentos/ensayo_rate_limit.py` | Provoca un límite de uso real contra el proveedor de respaldo y muestra el backoff y su techo en vivo. |
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
- Modelar los feriados del proveedor al calcular la tarifa de hora punta.
- Registrar el costo en un archivo o base de datos para conservarlo entre ejecuciones.

## 12. Conclusión

El proyecto integra un banco de tres prompts especializados con una API de inteligencia artificial generativa para analizar comentarios de encuestas de satisfacción.

La ejecución principal permite comprobar el funcionamiento del flujo de consultas, mientras que el script de consistencia amplía las pruebas mediante múltiples intentos, validación estructural y registro de resultados.

Las ejecuciones realizadas alcanzaron un 100 % de éxito en ambos procesos. Estos resultados permiten verificar el funcionamiento de las validaciones implementadas en los casos evaluados, sin garantizar la corrección semántica de todas las respuestas futuras.

El proyecto constituye una base para continuar desarrollando y evaluando soluciones de análisis de encuestas asistidas por inteligencia artificial.