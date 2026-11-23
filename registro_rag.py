"""Instrumentación de latencia y costo del flujo RAG completo (Bloque 28).

Qué mide
--------
Cada consulta se descompone en dos etapas y se cronometra por separado:

- ``embed_y_busqueda``: convertir la pregunta en embedding y buscar en el índice.
- ``generacion``: la llamada al modelo.

Sin ese desglose, un flujo lento no dice dónde está el cuello de botella: se
sabe que tarda, no qué arreglar. Con el desglose, la decisión es concreta.

Qué registra
------------
Una línea de JSON por consulta, con marca de tiempo, pregunta, ``k``, tiempos
por etapa, tiempo total, tokens reales informados por el proveedor y el costo
calculado con las tarifas por componente de ``src/costos.py``.

Por qué el costo se calcula con el módulo del proyecto y no con una tarifa única
-------------------------------------------------------------------------------
El material usa un solo precio por mil tokens porque su proveedor cobra una
tarifa simple. DeepSeek no: entrada y salida tienen precios distintos, la
entrada que acierta la caché cuesta cincuenta veces menos, y la hora punta
duplica la tarifa. Un único número habría quedado equivocado hasta por un factor
de trece. Reusar `costos.calcular_costo` mide el costo real del flujo real.

Formato del archivo
-------------------
JSON Lines: un objeto JSON por línea, sin lista contenedora. Se puede leer un
registro sin parsear el archivo entero, y una corrida nueva se agrega al final
sin reescribir lo anterior.

Uso
---
    ./.venv/Scripts/python.exe registro_rag.py
    ./.venv/Scripts/python.exe registro_rag.py --preguntas 5 --k 3
    ./.venv/Scripts/python.exe registro_rag.py --resumen
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.language_models.chat_models import BaseChatModel

RAIZ = Path(__file__).resolve().parent
SRC = RAIZ / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import indexar_corpus as ic  # noqa: E402
from costos import calcular_costo  # noqa: E402
from evaluacion_recuperacion import CASOS_EVALUACION  # noqa: E402
from recuperacion import K_POR_DEFECTO, construir_modelo, construir_prompt  # noqa: E402

logger = logging.getLogger(__name__)

ARCHIVO_REGISTROS = Path("registros_latencia_costo.jsonl")
CONSULTAS_POR_DEFECTO = 3


@contextmanager
def medir_tiempo(tiempos: dict, etapa: str):
    """Cronometra el bloque y guarda los milisegundos en ``tiempos[etapa]``."""

    inicio = time.perf_counter()
    try:
        yield
    finally:
        # En ``finally`` y no después del ``yield``: si la etapa falla, el tiempo
        # transcurrido igual sirve para saber cuánto se esperó antes del error.
        tiempos[etapa] = time.perf_counter() - inicio


def normalizar_uso(mensaje) -> dict:
    """Traduce ``usage_metadata`` de LangChain al bloque ``usage`` del proveedor.

    `costos.calcular_costo` espera las claves nativas de DeepSeek
    (``prompt_tokens``, ``completion_tokens`` y el desglose de caché). LangChain
    las entrega con otro nombre, así que la traducción se hace acá, en un solo
    lugar, en vez de repartirla por el código.

    Si el proveedor no informa el uso, devuelve un diccionario vacío: es una
    señal real de que esa consulta no trajo datos, no un bug de la medición.
    """

    metadatos = getattr(mensaje, "usage_metadata", None) or {}
    if not metadatos:
        return {}

    detalles = metadatos.get("input_token_details") or {}
    entrada = int(metadatos.get("input_tokens") or 0)
    salida = int(metadatos.get("output_tokens") or 0)
    cache_leida = int(detalles.get("cache_read") or 0)

    return {
        "prompt_tokens": entrada,
        "completion_tokens": salida,
        "prompt_cache_hit_tokens": cache_leida,
        # Lo no cacheado se cobra como miss, que es la tarifa más cara: nunca
        # subestimar el costo por falta de un dato.
        "prompt_cache_miss_tokens": max(entrada - cache_leida, 0),
        "total_tokens": int(metadatos.get("total_tokens") or entrada + salida),
    }


def calcular_costo_de_uso(uso: dict, modelo: str, momento: datetime | None = None) -> float:
    """Costo en USD a partir del uso normalizado. 0.0 si no hay datos de uso."""

    if not uso:
        return 0.0
    return calcular_costo({"usage": uso}, modelo=modelo, momento=momento)


def registrar_consulta(
    pregunta: str,
    vectorstore: Chroma,
    modelo: BaseChatModel,
    modelo_cobrado: str,
    k: int = K_POR_DEFECTO,
    momento: datetime | None = None,
) -> dict:
    """Corre una consulta RAG real y devuelve su registro completo.

    Ejecuta exactamente el mismo camino que ``recuperacion.responder``, para que
    lo medido sea el flujo del proyecto y no una copia paralela que se
    desactualiza al primer cambio.
    """

    tiempos: dict[str, float] = {}

    with medir_tiempo(tiempos, "embed_y_busqueda"):
        prompt, citas = construir_prompt(pregunta, vectorstore, k=k)

    with medir_tiempo(tiempos, "generacion"):
        respuesta = modelo.invoke(prompt)

    uso = normalizar_uso(respuesta)
    ahora = momento or datetime.now(timezone.utc)

    return {
        "timestamp": ahora.isoformat(),
        "pregunta": pregunta,
        "k": k,
        "tiempos_ms": {etapa: round(segundos * 1000, 2) for etapa, segundos in tiempos.items()},
        "tiempo_total_ms": round(sum(tiempos.values()) * 1000, 2),
        "tokens": uso or None,
        "costo_usd": round(calcular_costo_de_uso(uso, modelo_cobrado, ahora), 8),
        "modelo": modelo_cobrado,
        "fuentes": sorted(set(citas.values())),
    }


def agregar_registro(registro: dict, ruta: Path = ARCHIVO_REGISTROS) -> None:
    """Agrega una línea al archivo de registros, sin reescribir las anteriores."""

    with ruta.open("a", encoding="utf-8") as archivo:
        archivo.write(json.dumps(registro, ensure_ascii=False) + "\n")


def leer_registros(ruta: Path = ARCHIVO_REGISTROS) -> list[dict]:
    """Lee el archivo completo. Lanza ``FileNotFoundError`` si no existe."""

    with ruta.open(encoding="utf-8") as archivo:
        return [json.loads(linea) for linea in archivo if linea.strip()]


def resumir_registros(registros: list[dict]) -> str:
    """Devuelve un resumen legible: una línea por corrida y el total acumulado."""

    if not registros:
        return "Sin registros"

    lineas = []
    for registro in registros:
        tokens = registro.get("tokens") or {}
        entrada = tokens.get("prompt_tokens", "n/a")
        lineas.append(
            f"k={registro['k']}  entrada={entrada}  "
            f"total={registro['tiempo_total_ms']} ms  "
            f"${registro.get('costo_usd', 0):.8f}  {registro['pregunta'][:60]}"
        )

    total = sum(registro.get("costo_usd", 0) for registro in registros)
    lineas.append(f"\n{len(registros)} corridas  costo total: ${total:.8f}")
    return "\n".join(lineas)


def preguntas_de_ejemplo(cantidad: int) -> list[str]:
    """Toma las primeras preguntas respondibles del conjunto de evaluación.

    Reusar el conjunto del Bloque 27 evita inventar consultas nuevas cada vez: la
    latencia se mide sobre las mismas preguntas que ya se evaluaron por
    recuperación, así los dos números describen el mismo flujo.
    """

    respondibles = [
        caso["pregunta"] for caso in CASOS_EVALUACION if caso["ids_relevantes"]
    ]
    return respondibles[:cantidad]


def main(argumentos: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    load_dotenv()

    parser = argparse.ArgumentParser(description="Registra latencia y costo del flujo RAG.")
    parser.add_argument("--preguntas", type=int, default=CONSULTAS_POR_DEFECTO)
    parser.add_argument("--k", type=int, default=K_POR_DEFECTO)
    parser.add_argument("--resumen", action="store_true", help="solo muestra los registros guardados")
    opciones = parser.parse_args(argumentos)

    if opciones.resumen:
        try:
            print(resumir_registros(leer_registros()))
        except FileNotFoundError:
            print(f"Todavía no hay registros en {ARCHIVO_REGISTROS}")
            return 1
        return 0

    try:
        embeddings = ic.construir_embeddings()
        vectorstore = ic.abrir_indice(embeddings)
        ic.verificar_indice(vectorstore)
    except ic.ErrorCorpus as error:
        print(f"ERROR: {error}")
        return 1

    modelo = construir_modelo()
    modelo_cobrado = os.environ.get("DEEPSEEK_MODEL", "deepseek-flash")

    for pregunta in preguntas_de_ejemplo(opciones.preguntas):
        registro = registrar_consulta(
            pregunta, vectorstore, modelo, modelo_cobrado, k=opciones.k
        )
        agregar_registro(registro)
        print(json.dumps(registro, indent=2, ensure_ascii=False))
        print()

    print(resumir_registros(leer_registros()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
