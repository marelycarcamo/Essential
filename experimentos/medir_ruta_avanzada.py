"""Mide la ruta avanzada contra cada proveedor: cuánto tarda y qué lanza.

Motivo: la primera corrida real de ``enrutamiento.py avanzado`` falló y el
respaldo tapó el error del principal. Este script mide una llamada por
proveedor, sin cadena de respaldo, para ver el error tal cual.

Mide, además, dos cosas que la documentación del curso no dice:

- ``timeout`` en ``ChatGoogleGenerativeAI`` es **por intento**, y el cliente
  reintenta por su cuenta. El techo real es ``timeout × (1 + max_retries)``.
- El respaldo adaptado (``adaptar_respaldo``) tiene que funcionar de verdad,
  no solo en las pruebas con dobles.

Uso:
    .\\.venv\\Scripts\\python.exe experimentos\\medir_ruta_avanzada.py gemini
    .\\.venv\\Scripts\\python.exe experimentos\\medir_ruta_avanzada.py deepseek
"""

import logging
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))

from cadena_tres_pasos import TEXTO_DE_EJEMPLO  # noqa: E402
from enrutamiento import PLANTILLA_AVANZADA, adaptar_respaldo  # noqa: E402
from src.deepseek_wrapper import DeepSeekWrapper  # noqa: E402

TIEMPO_MAXIMO_SEGUNDOS = 90.0
REINTENTOS = 0

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("medir_ruta_avanzada")


def construir_gemini() -> tuple[object, str]:
    """Modelo de Gemini con techo acotado: sin reintentos ocultos."""

    modelo = ChatGoogleGenerativeAI(
        model=os.getenv("GEMINI_MODEL", "gemini-3.8-flash"),
        google_api_key=os.getenv("GEMINI_API_KEY"),
        temperature=0,
        timeout=TIEMPO_MAXIMO_SEGUNDOS,
        max_retries=REINTENTOS,
    )

    return modelo, f"{modelo.model} (timeout {TIEMPO_MAXIMO_SEGUNDOS:.0f} s × retries {REINTENTOS})"


def construir_deepseek() -> tuple[object, str]:
    """Respaldo adaptado, exactamente como lo usa la cadena segura."""

    wrapper = DeepSeekWrapper()

    return adaptar_respaldo(wrapper), f"{wrapper.model} (timeout del wrapper)"


def main() -> int:
    """Mide una llamada y reporta latencia, tipo de error y largo del texto."""

    load_dotenv(RAIZ / ".env")

    proveedor = sys.argv[1] if len(sys.argv) > 1 else "gemini"

    if proveedor == "gemini":
        modelo, descripcion = construir_gemini()
    elif proveedor == "deepseek":
        modelo, descripcion = construir_deepseek()
    else:
        logger.error("Proveedor desconocido: %s (usa gemini o deepseek)", proveedor)
        return 2

    logger.info("Proveedor: %s", descripcion)
    logger.info("Plantilla: %d caracteres", len(PLANTILLA_AVANZADA.template))

    inicio = time.monotonic()

    try:
        resultado = modelo.invoke(
            PLANTILLA_AVANZADA.invoke({"texto": TEXTO_DE_EJEMPLO})
        )
    except Exception as error:  # noqa: BLE001 - queremos ver cualquier fallo
        logger.error(
            "FALLA a los %.1f s · %s · %s",
            time.monotonic() - inicio,
            type(error).__name__,
            str(error)[:400],
        )
        return 1

    salida = getattr(resultado, "content", resultado)

    logger.info("OK a los %.1f s", time.monotonic() - inicio)
    logger.info("tipo: %s", type(resultado).__name__)
    logger.info("largo: %d caracteres", len(str(salida)))
    logger.info("inicio del texto: %s", str(salida)[:120].replace("\n", " "))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
