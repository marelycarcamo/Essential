"""¿Qué proveedor de nuestro proyecto sabe pedir herramientas?

Motivo: los Bloques 21 a 23 giran alrededor de un agente con herramientas, y un
agente necesita un modelo que sepa *pedir* herramientas, no solo escribir texto.

Este script gasta **una** llamada por proveedor: une una herramienta trivial al
modelo y verifica si la pide de verdad.

Uso:
    .\\.venv\\\\Scripts\\\\python.exe experimentos\\\\probar_tool_calling.py deepseek
    .\\.venv\\\\Scripts\\\\python.exe experimentos\\\\probar_tool_calling.py gemini
"""

import logging
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.tools import tool

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("probar_tool_calling")

PROVEEDORES_VALIDOS = ("deepseek", "gemini")


@tool
def sumar(a: int, b: int) -> str:
    """Suma dos números enteros. Usar cuando se pide una suma."""

    return str(a + b)


def construir_modelo_deepseek():
    """Modelo de DeepSeek por el camino oficial de LangChain."""

    from langchain_deepseek import ChatDeepSeek

    if not os.getenv("DEEPSEEK_API_KEY"):
        raise RuntimeError("Falta DEEPSEEK_API_KEY en el entorno")

    return ChatDeepSeek(
        model=os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
        temperature=0,
        timeout=60,
        max_retries=0,
    )


def construir_modelo_gemini():
    """Modelo de Gemini reusando la fábrica del proyecto."""

    from enrutamiento import construir_modelo_gemini as fabrica

    clave = os.getenv("GEMINI_API_KEY")
    if not clave:
        raise RuntimeError("Falta GEMINI_API_KEY en el entorno")

    return fabrica(clave)


FABRICAS = {"deepseek": construir_modelo_deepseek, "gemini": construir_modelo_gemini}


def main() -> int:
    """Pregunta algo que solo se responde usando la herramienta."""

    load_dotenv(RAIZ / ".env")

    proveedor = sys.argv[1] if len(sys.argv) > 1 else "deepseek"
    if proveedor not in PROVEEDORES_VALIDOS:
        logger.error("Proveedor desconocido: %s", proveedor)
        return 2

    modelo = FABRICAS[proveedor]().bind_tools([sumar])

    inicio = time.monotonic()

    try:
        respuesta = modelo.invoke("¿Cuánto es 2 + 3? Usa la herramienta sumar.")
    except Exception as error:  # noqa: BLE001 - queremos ver cualquier fallo
        logger.error(
            "FALLA a los %.1f s · %s · %s",
            time.monotonic() - inicio,
            type(error).__name__,
            str(error)[:300],
        )
        return 1

    logger.info("%s · latencia: %.1f s", proveedor, time.monotonic() - inicio)
    logger.info("pidió herramientas: %s", bool(respuesta.tool_calls))
    logger.info("tool_calls: %s", respuesta.tool_calls)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
