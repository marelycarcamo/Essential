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
│   ├── comparar_modelos.py
│   ├── consumir_deepseek.py
│   ├── consumir_openrouter.py
│   └── parsear_json.py
├── tests/
│   ├── conftest.py
│   ├── test_banco_prompts.py
│   ├── test_consumir_deepseek.py
│   ├── test_consumir_openrouter.py
│   ├── test_integracion.py
│   ├── test_main.py
│   └── test_parsear_json.py
├── .env.example
├── .gitignore
├── main.py
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
| `src/parsear_json.py` | Procesa las respuestas JSON y comprueba las claves esperadas. |
| `prompts/banco.py` | Define los prompts y sus esquemas para las pruebas de consistencia. |
| `prompts/test_consistencia.py` | Ejecuta los prompts varias veces, valida las respuestas y registra los resultados. |
| `prompts/registro_resultados.json` | Almacena los resultados de la última ejecución de las pruebas de consistencia. |
| `notebooks/prueba_banco_prompts.ipynb` | Permite experimentar y evaluar los prompts durante el desarrollo. |

Los archivos `src/banco_prompts.py` y `prompts/banco.py` son independientes. Actualmente, `main.py` utiliza el primero, mientras que el script de consistencia utiliza el segundo.

## 6. Requisitos previos

Se necesita:

- Python instalado.
- Una clave API de OpenRouter.
- Las dependencias declaradas en `requirements.txt`.

Instala las dependencias desde la raíz del repositorio:

```bash
pip install -r requirements.txt
```

Si trabajas con un entorno virtual, actívalo antes de ejecutar los comandos.

## 7. Configuración

Crea un archivo `.env` en la raíz del proyecto con las siguientes variables:

```dotenv
OPENROUTER_API_KEY=tu_clave_api
OPENROUTER_MODEL=openai/gpt-4o-mini
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1/chat/completions
```

Reemplaza `tu_clave_api` por tu propia clave de OpenRouter.

**Importante:** No compartas tu clave API ni la subas al repositorio. El archivo `.env` debe estar excluido mediante `.gitignore`. El archivo `.env.example` puede documentar las variables necesarias sin incluir credenciales reales.

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

## 10. Registro del desarrollo y ensayo de instalación limpia

- **Fecha:** 2026-10-23
- **Desarrollo:** creación del banco de prompts, elaboración de un notebook y preparación de pruebas automatizadas.
- **Pruebas:** implementación de pruebas unitarias, archivo `conftest.py` para la configuración compartida de las pruebas y pruebas de consistencia de los prompts.
- **Hallazgo:** después de publicar el Bloque 10, descubrí que faltaba incluir `pytest` en `requirements.txt`, porque había incorporado las pruebas posteriormente.
- **Corrección:** actualicé `requirements.txt` y registré el cambio en Git antes de iniciar el Bloque 11.

### Bloque 11: Ensayo de instalación limpia

- **Procedimiento:** cloné mi repositorio, instalé las dependencias y configuré las variables de entorno mediante el archivo `.env`.
- **Resultado:** ejecuté `main.py` sin errores en el repositorio clonado.
- **Validación de consistencia:** ejecuté `prompts/test_consistencia.py` y obtuve 9 de 9 respuestas válidas (100 % de éxito).
- **Evidencia:** notebook del proyecto, pruebas automatizadas, `conftest.py` y archivo `prompts/registro_resultados.json` generado durante las pruebas.

## 12. Conclusión

El proyecto integra un banco de tres prompts especializados con una API de inteligencia artificial generativa para analizar comentarios de encuestas de satisfacción.

La ejecución principal permite comprobar el funcionamiento del flujo de consultas, mientras que el script de consistencia amplía las pruebas mediante múltiples intentos, validación estructural y registro de resultados.

Las ejecuciones realizadas alcanzaron un 100 % de éxito en ambos procesos. Estos resultados permiten verificar el funcionamiento de las validaciones implementadas en los casos evaluados, sin garantizar la corrección semántica de todas las respuestas futuras.

El proyecto constituye una base para continuar desarrollando y evaluando soluciones de análisis de encuestas asistidas por inteligencia artificial.