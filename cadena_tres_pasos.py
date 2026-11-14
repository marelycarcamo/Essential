"""Chain de tres pasos con LangChain — Bloque 18.

Caso: ALESSIA (proyecto previo) — convertir la documentación técnica de su
pipeline RAG en material de capacitación para un curso. El texto de entrada
es real, no un párrafo de prueba.

Por qué tres pasos y no un prompt largo partido en tres
-------------------------------------------------------
Cada paso hace un **tipo de trabajo distinto** y recibe algo que el anterior
no tenía:

1. **Analizar** — decide qué clase de documento es, para qué nivel sirve y qué
   tres conceptos enseña. Es una decisión: salida corta, de formato fijo y con
   ``temperature=0``. No redacta nada.
2. **Planificar** — con la ficha del paso 1, diseña la sesión: objetivo de
   aprendizaje, conceptos en orden de dificultad y un ejercicio. Tampoco
   redacta el contenido: decide su estructura.
3. **Redactar** — recién aquí se escribe el material del alumno, siguiendo el
   plan. Es redacción pura, con el nivel y el orden ya resueltos.

La diferencia con un ``prompt`` que pida todo junto: si el modelo tiene que
decidir el nivel, ordenar los conceptos y redactar en la misma llamada, se
salta la decisión y escribe lo que le resulta más fácil. Separado, cada paso
responde una sola pregunta.

Qué NO resuelve
---------------
- **No consulta los documentos de ALESSIA.** El texto entra escrito a mano en
  ``TEXTO_DE_EJEMPLO``. En el pipeline real esa recuperación ocurre antes.
- **No conserva el texto original.** Después de un ``StrOutputParser`` viaja un
  string, así que el paso 2 trabaja con la ficha y no con el documento. Para
  arrastrar datos de la entrada existe ``RunnablePassthrough.assign``, que es
  materia del Bloque 19.
- **No enruta.** Los tres pasos corren siempre, en el mismo orden.

Proyecto: ESSENTIAL — actividad práctica del Bloque 18.
"""

from __future__ import annotations

import logging
import os
import unicodedata

import requests
from dotenv import load_dotenv
from google.genai.errors import APIError as ErrorDelProveedor
from langchain_core.exceptions import (
    ModelAPIError,
    ModelAuthenticationError,
    ModelRateLimitError,
)
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import Runnable, RunnableLambda
from langchain_google_genai import ChatGoogleGenerativeAI

from src.deepseek_wrapper import DeepSeekWrapper

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# Errores que justifican cambiar de modelo
# --------------------------------------------------------------------------

# Solo entran aquí los fallos del proveedor: saturación, cuota, red caída.
# Un KeyError o un TypeError significan que la cadena está mal armada, y en
# ese caso hay que verlos, no rotar de modelo hasta que el mensaje se pierda.
#
# Ojo con las dos familias: LangChain **normaliza** una parte de los errores de
# Google en sus propias clases, que no heredan de ``google.genai.APIError``. El
# 429 es el caso que nos mordió: llegaba como ``GoogleRateLimitError``, no
# estaba en la lista y se escapaba del respaldo.
ERRORES_DE_PROVEEDOR = (
    ErrorDelProveedor,         # 5xx tal cual vienen de google.genai
    ModelAPIError,             # 5xx normalizados por LangChain
    ModelRateLimitError,       # 429: se agotó la cuota, no el servicio
    ModelAuthenticationError,  # clave inválida: el respaldo es la salida
    requests.RequestException,
    TimeoutError,
    ConnectionError,
)

# Deliberadamente **fuera** de la lista: ``ModelNotFoundError`` (un 404 es un
# modelo mal escrito en la configuración, como el ``gemini-3-flash`` que nadie
# notó durante un bloque entero) y ``ModelInvalidRequestError`` (un 400 es
# nuestro prompt mal armado). Rotar de modelo ante esos dos esconde el error en
# vez de arreglarlo.


# --------------------------------------------------------------------------
# Paso 1 — analizar el insumo
# --------------------------------------------------------------------------

CAMPOS_FICHA = ("tipo", "nivel", "conceptos", "requisito")
NIVELES_VALIDOS = ("inicial", "intermedio", "avanzado")
NIVEL_POR_DEFECTO = "intermedio"

PLANTILLA_ANALISIS = PromptTemplate.from_template(
    "Eres diseñadora instruccional. Antes de escribir material de "
    "capacitación sobre un documento técnico, necesitas una ficha del "
    "insumo.\n\n"
    "Lee el documento y devuelve SOLO una línea con cuatro campos separados "
    "por el carácter |, en este orden y sin texto adicional:\n\n"
    "tipo|nivel|conceptos|requisito\n\n"
    "- tipo: qué clase de documento es (README, guía, docstring, artículo).\n"
    "- nivel: una sola palabra entre inicial, intermedio o avanzado.\n"
    "- conceptos: los tres conceptos que enseña el documento, separados por "
    "punto y coma.\n"
    "- requisito: qué debería saber el alumno antes de esta sesión, en una "
    "frase.\n\n"
    "Documento:\n"
    "{texto}"
)


def sin_tildes(texto: str) -> str:
    """Devuelve el texto en minúsculas y sin tildes.

    El modelo escribe el nivel como lo lee: «Avanzado». Compararlo contra el
    catálogo sin normalizar haría que un valor correcto cayera al default, y
    el material saldría con el nivel equivocado sin que nadie lo note.

    Es pública a propósito: el enrutamiento del Bloque 19 compara contra el
    mismo catálogo y reusar esta función evita que las dos normalicen distinto.
    """

    descompuesto = unicodedata.normalize("NFKD", texto.strip().lower())

    return "".join(
        caracter
        for caracter in descompuesto
        if not unicodedata.combining(caracter)
    )


def normalizar_opcion(
    valor: str,
    permitidas: tuple[str, ...],
    por_defecto: str,
) -> str:
    """Devuelve ``valor`` normalizado si está en el catálogo; si no, el default."""

    limpio = sin_tildes(valor)

    return limpio if limpio in permitidas else por_defecto


def interpretar_ficha(texto: str) -> dict[str, str]:
    """Convierte la línea del paso 1 en el diccionario que espera el paso 2.

    Args:
        texto: línea con los cuatro campos separados por ``|``.

    Returns:
        Diccionario con las claves de ``CAMPOS_FICHA``.

    Si el modelo devuelve menos campos de los pedidos o un nivel que no está
    en el catálogo, se completa con valores seguros en vez de lanzar una
    excepción: un material con nivel intermedio por defecto se puede usar, pero
    una excepción deja la sesión sin material.
    """

    partes = [parte.strip() for parte in texto.split("|")]
    partes += [""] * (len(CAMPOS_FICHA) - len(partes))

    return {
        "tipo": partes[0] or "documento técnico",
        "nivel": normalizar_opcion(partes[1], NIVELES_VALIDOS, NIVEL_POR_DEFECTO),
        "conceptos": partes[2] or "no especificados",
        "requisito": "|".join(partes[3:]).strip() or "ninguno en particular",
    }


def construir_paso_analisis(modelo: Runnable) -> Runnable:
    """Devuelve el paso que convierte el documento en una ficha del insumo."""

    return (
        PLANTILLA_ANALISIS
        | modelo
        | StrOutputParser()
        | interpretar_ficha
    )


# --------------------------------------------------------------------------
# Paso 2 — diseñar la sesión
# --------------------------------------------------------------------------

PLANTILLA_PLAN = PromptTemplate.from_template(
    "Eres diseñadora instruccional. Ya tienes la ficha del insumo técnico:\n\n"
    "- Tipo de documento: {tipo}\n"
    "- Nivel del público: {nivel}\n"
    "- Conceptos que enseña: {conceptos}\n"
    "- Requisito previo: {requisito}\n\n"
    "Diseña una sesión de capacitación de veinte minutos. Devuelve:\n"
    "1. El objetivo de aprendizaje en una frase, empezando con un verbo de "
    "acción.\n"
    "2. Los tres conceptos en orden de dificultad, cada uno con una analogía "
    "breve.\n"
    "3. Un ejercicio práctico de una línea.\n\n"
    "No expliques el contenido a fondo: eso es la sesión, no el plan."
)


def construir_paso_plan(modelo: Runnable) -> Runnable:
    """Devuelve el paso que arma el plan de la sesión a partir de la ficha."""

    return PLANTILLA_PLAN | modelo | StrOutputParser()


# --------------------------------------------------------------------------
# Paso 3 — redactar el material
# --------------------------------------------------------------------------

PLANTILLA_MATERIAL = PromptTemplate.from_template(
    "Toma este plan de sesión y escribe el material del alumno:\n\n"
    "{plan}\n\n"
    "Reglas:\n"
    "- Una sección por concepto, con un título corto.\n"
    "- Explica en el nivel indicado en el plan y define cada término técnico "
    "la primera vez que aparece.\n"
    "- Cierra con el ejercicio.\n"
    "- Máximo doscientas palabras.\n"
    "- No agregues datos, cifras ni nombres que no estén en el plan."
)


def empaquetar_plan(plan: str) -> dict[str, str]:
    """Envuelve el texto del paso 2 en el diccionario que espera el paso 3.

    Un ``PromptTemplate`` recibe siempre un diccionario, pero después del
    ``StrOutputParser`` lo que viaja es un string. Esta función es la bisagra
    entre los dos formatos, y es el error más común de la actividad: si la
    clave no coincide con la variable del template, la cadena muere con
    ``KeyError`` en el último paso.
    """

    return {"plan": plan}


def construir_paso_material(modelo: Runnable) -> Runnable:
    """Devuelve el paso que redacta el material del alumno."""

    return PLANTILLA_MATERIAL | modelo | StrOutputParser()


# --------------------------------------------------------------------------
# La cadena completa
# --------------------------------------------------------------------------


def construir_cadena(modelo: Runnable) -> Runnable:
    """Arma los tres pasos y devuelve la cadena **sin ejecutarla**.

    Recibe el modelo como argumento, y no lo crea aquí dentro, por dos
    razones: las pruebas pueden inyectar un modelo falso y no gastar cuota, y
    cambiar de proveedor no obliga a tocar la cadena.
    """

    return (
        construir_paso_analisis(modelo)
        | construir_paso_plan(modelo)
        | empaquetar_plan
        | construir_paso_material(modelo)
    )


def invocar_con_conmutador(
    modelo_principal: Runnable,
    modelos_alternativos: list[Runnable] | None,
    inputs: dict[str, str],
) -> str:
    """Invoca la cadena probando modelos en orden hasta que uno responda.

    Sirve para los fallos que no dependen de nosotros: cuota agotada (429),
    servicio saturado (503) o red caída. **No** atrapa errores de programación:
    si la cadena está mal armada, la excepción sube tal cual, porque rotar de
    modelo no arregla un ``KeyError`` y esconderlo convierte un error de cinco
    segundos en una tarde de diagnóstico.

    Args:
        modelo_principal: primer modelo a intentar.
        modelos_alternativos: respaldos, en el orden en que se probarán.
        inputs: diccionario de entrada de la cadena.

    Returns:
        El texto devuelto por el primer modelo que responda.

    Raises:
        RuntimeError: si todos los modelos fallaron. Encadena la última causa
            con ``from`` para no perder el motivo original.
    """

    ultimo_error: Exception | None = None

    for modelo in [modelo_principal, *(modelos_alternativos or [])]:
        try:
            return construir_cadena(modelo).invoke(inputs)
        except ERRORES_DE_PROVEEDOR as error:
            ultimo_error = error
            logger.warning(
                "Un modelo falló (%s: %s). Se prueba el siguiente.",
                type(error).__name__,
                error,
            )

    raise RuntimeError(
        "Ningún modelo pudo responder la cadena"
    ) from ultimo_error


# --------------------------------------------------------------------------
# Ejecución directa
# --------------------------------------------------------------------------

# Texto real: describe el pipeline RAG que ALESSIA construyó. Se usa como
# insumo de la capacitación, no como ejemplo de laboratorio.
TEXTO_DE_EJEMPLO = (
    "ALESSIA is a RAG-based chatbot built with Python, LangGraph, and ChromaDB. "
    "It ingests PDFs, splits them into chunks with overlap, embeds them using "
    "a multilingual model, stores them in a vector index, retrieves top-k fragments, "
    "reranks them with a CrossEncoder, and verifies fidelity before generating answers. "
    "The pipeline uses structured output (Pydantic) and persists state in OCI."
)


def configurar_registro() -> None:
    """Configura el registro del script, silenciando el ruido de las librerías."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(message)s",
    )

    # Subir el nivel global también sube el de las librerías: httpx y el SDK de
    # Google registran cada petición en INFO y llenan la salida de ruido ajeno.
    for ruidoso in ("httpx", "httpcore", "google_genai", "google.auth", "urllib3"):
        logging.getLogger(ruidoso).setLevel(logging.WARNING)


def main() -> int:
    """Ejecuta la cadena contra los proveedores reales."""

    configurar_registro()
    load_dotenv()

    clave = os.getenv("GEMINI_API_KEY")
    if not clave:
        logger.error(
            "Falta GEMINI_API_KEY en el entorno. Se espera en el .env del "
            "proyecto; sin ella no hay llamada al modelo."
        )
        return 1

    # El modelo se crea aquí y no al importar el módulo: importar no debe
    # exigir credenciales ni abrir clientes HTTP. Si estuviera arriba, un clon
    # sin .env fallaría antes de que corra el primer test.
    modelo_principal = ChatGoogleGenerativeAI(
        model=os.getenv("GEMINI_MODEL", "gemini-3.8-flash"),
        google_api_key=clave,
        temperature=0,
    )

    alternativos: list[Runnable] = []
    try:
        # ``DeepSeekWrapper`` no es un Runnable ni un callable: expone
        # ``invoke`` pero LangChain exige el protocolo completo para componerlo
        # con ``|``. Sin este adaptador, la cadena ni se arma y el respaldo
        # queda como código muerto que parece funcionar.
        alternativos.append(RunnableLambda(DeepSeekWrapper().invoke))
    except RuntimeError as error:
        logger.info("Sin modelo de respaldo (%s). Se usa solo el principal.", error)

    resultado = invocar_con_conmutador(
        modelo_principal,
        alternativos,
        {"texto": TEXTO_DE_EJEMPLO},
    )

    print("=== Material de capacitación generado en tres pasos ===")
    print(resultado)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
