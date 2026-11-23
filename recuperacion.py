"""Recuperación semántica y prompt dinámico con citas (Bloque 26).

Qué hace
--------
Recibe una pregunta, busca en el índice vectorial los fragmentos más parecidos
por significado, los numera y arma con ellos un prompt que obliga al modelo a
citar la fuente de cada afirmación. Devuelve también el mapa de citas, para
poder comprobar después que el número citado apunta al documento correcto.

Por qué no reindexar aquí
-------------------------
Esta capa se reconecta al índice con ``Chroma(persist_directory=...)``. Si
volviera a llamar a ``from_documents``, recalcularía los embeddings que ya están
en disco en **cada pregunta**: más lento, más caro y sin ninguna ventaja.

Por qué citar
-------------
Una respuesta sin fuente no se puede auditar. El número entre corchetes no es
decoración: permite ir del texto a un archivo concreto del corpus y confirmar o
desmentir la afirmación. Por eso el prompt incluye además la instrucción de
admitir cuando el contexto no alcanza, que es la otra mitad de la garantía: un
sistema que siempre responde algo no distingue entre saber y no saber.

Uso
---
    ./.venv/Scripts/python.exe recuperacion.py "¿qué pasa si el proveedor falla?"

Requiere el índice creado con ``python indexar_corpus.py``.
"""

from __future__ import annotations

import logging
import os
import re
import sys

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_deepseek import ChatDeepSeek

import indexar_corpus as ic

logger = logging.getLogger(__name__)

# Una cita es un número entre corchetes. Se expresa como patrón y no como un
# recorrido carácter por carácter porque el recorrido se rompe con un corchete
# de cierre suelto, y ese error no avisa.
PATRON_CITA = re.compile(r"\[(\d+)\]")

# k=3 no es un número elegido al azar ni el máximo posible. Con 34 fragmentos de
# un párrafo, tres cubren la pregunta y su contexto inmediato; subirlo mete
# fragmentos débilmente relacionados que diluyen la atención del modelo, y
# bajarlo a uno pierde el contexto que la cita necesita. El Bloque 27 mide este
# trade-off con precisión y recall en lugar de confiarlo a la intuición.
K_POR_DEFECTO = 3

# La respuesta no puede salir del contexto, y cuando el contexto no alcanza debe
# decirlo. Las dos instrucciones están juntas porque una sola no alcanza: "cita
# tus fuentes" sin "admite cuando no sabes" produce citas inventadas.
PLANTILLA_PROMPT = """Responde la pregunta usando SOLO el contexto de abajo.
Cita la fuente de cada afirmación con su número entre corchetes, como [1].
Si el contexto no alcanza para responder, dilo explícitamente en vez de inventar.

Contexto:
{contexto}

Pregunta: {pregunta}

Respuesta:"""


class ErrorConsulta(ValueError):
    """La pregunta no es válida o el índice no se puede consultar."""


def formatear_contexto(fragmentos: list[Document]) -> tuple[str, dict[int, str]]:
    """Numera los fragmentos y arma el mapa de citas.

    Args:
        fragmentos: fragmentos recuperados, en el orden en que los devolvió el
            retriever (del más al menos parecido).

    Returns:
        Una tupla con el bloque de contexto numerado y un diccionario que va del
        número citado al archivo del que salió el fragmento.
    """

    lineas: list[str] = []
    citas: dict[int, str] = {}

    for numero, fragmento in enumerate(fragmentos, start=1):
        fuente = fragmento.metadata.get("fuente", "desconocida")
        lineas.append(f"[{numero}] {fragmento.page_content.strip()}")
        citas[numero] = fuente

    return "\n\n".join(lineas), citas


def validar_pregunta(pregunta: str) -> str:
    """Comprueba que la pregunta sea texto con contenido y la devuelve limpia.

    Raises:
        ErrorConsulta: si no es texto, si está vacía o si solo tiene espacios.
    """

    if not isinstance(pregunta, str):
        raise ErrorConsulta(
            f"La pregunta debe ser texto, no {type(pregunta).__name__}"
        )
    limpia = pregunta.strip()
    if not limpia:
        raise ErrorConsulta("La pregunta está vacía")
    return limpia


def recuperar(
    vectorstore: Chroma,
    pregunta: str,
    k: int = K_POR_DEFECTO,
) -> list[Document]:
    """Trae los ``k`` fragmentos más parecidos por significado a la pregunta."""

    if k < 1:
        raise ErrorConsulta(f"k debe ser al menos 1, no {k}")

    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    fragmentos = retriever.invoke(pregunta)

    if not fragmentos:
        raise ErrorConsulta(
            "La búsqueda no devolvió fragmentos: revisa que el índice tenga contenido"
        )

    logger.info("Recuperados %d fragmentos para la consulta", len(fragmentos))
    return fragmentos


def construir_prompt(
    pregunta: str,
    vectorstore: Chroma,
    k: int = K_POR_DEFECTO,
) -> tuple[str, dict[int, str]]:
    """Recupera el contexto y arma el prompt dinámico con sus citas.

    Returns:
        El prompt listo para el modelo y el mapa de citas.
    """

    limpia = validar_pregunta(pregunta)
    fragmentos = recuperar(vectorstore, limpia, k=k)
    contexto, citas = formatear_contexto(fragmentos)
    return PLANTILLA_PROMPT.format(contexto=contexto, pregunta=limpia), citas


def citas_usadas(respuesta: str) -> list[int]:
    """Extrae los números de cita que aparecen en la respuesta, en orden de aparición."""

    return [int(numero) for numero in PATRON_CITA.findall(respuesta)]


def verificar_citas(respuesta: str, citas: dict[int, str]) -> list[str]:
    """Comprueba que las citas de la respuesta existan en el mapa.

    Informa, no bloquea: la tarea de esta función es convertir una respuesta
    aparentemente correcta en una lista de problemas revisables. Una cita a un
    número que no existe es una alucinación de formato, y es el modo de falla
    más común de este patrón.

    Returns:
        Lista de problemas. Vacía si todas las citas apuntan a un fragmento real.
    """

    if not citas:
        return ["No hay mapa de citas: la respuesta no se puede auditar"]

    usadas = citas_usadas(respuesta)
    problemas = [f"cita [{numero}] sin fragmento asociado" for numero in usadas if numero not in citas]

    if not usadas:
        problemas.append("la respuesta no cita ninguna fuente")

    return problemas


def construir_modelo() -> BaseChatModel:
    """Crea el modelo de chat del proyecto (DeepSeek) desde las variables de entorno."""

    if not os.environ.get("DEEPSEEK_API_KEY"):
        raise SystemExit(
            "Falta DEEPSEEK_API_KEY. Defínela en el archivo .env de la raíz del "
            "proyecto y vuelve a ejecutar."
        )
    return ChatDeepSeek(
        model=os.environ.get("DEEPSEEK_MODEL", "deepseek-flash"),
        timeout=60,
        max_retries=2,
    )


def responder(
    pregunta: str,
    vectorstore: Chroma,
    modelo: BaseChatModel,
    k: int = K_POR_DEFECTO,
) -> dict:
    """Corre la consulta completa: recuperar, armar el prompt y generar.

    Returns:
        Un diccionario con la pregunta, el prompt, el mapa de citas, la respuesta
        y los problemas de cita detectados.
    """

    prompt, citas = construir_prompt(pregunta, vectorstore, k=k)
    mensaje = modelo.invoke(prompt)
    texto = mensaje.content if isinstance(mensaje.content, str) else str(mensaje.content)

    return {
        "pregunta": pregunta,
        "prompt": prompt,
        "citas": citas,
        "respuesta": texto,
        "problemas_citas": verificar_citas(texto, citas),
    }


def main(argumentos: list[str]) -> int:
    """Responde una pregunta de la línea de comandos y muestra la trazabilidad."""

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    load_dotenv()

    if len(argumentos) < 2:
        print('Uso: python recuperacion.py "tu pregunta"')
        return 2

    pregunta = " ".join(argumentos[1:])

    try:
        embeddings = ic.construir_embeddings()
        vectorstore = ic.abrir_indice(embeddings)
        ic.verificar_indice(vectorstore)
        resultado = responder(pregunta, vectorstore, construir_modelo())
    except (ic.ErrorCorpus, ErrorConsulta) as error:
        print(f"ERROR: {error}")
        return 1

    print(f"\nPrompt enviado:\n{resultado['prompt']}\n")
    print(f"Respuesta:\n{resultado['respuesta']}\n")
    print(f"Citas: {resultado['citas']}")
    if resultado["problemas_citas"]:
        print(f"Problemas de cita: {resultado['problemas_citas']}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
