"""Verificación del entorno antes de la corrida final (Gate 2).

Qué comprueba
-------------
Lo que puede faltar en una máquina distinta de la de desarrollo: la versión de
Python, cada librería que el servicio importa, las claves de API y el índice
persistido. No imprime el valor de ninguna clave; solo si está definida.

Termina con código 1 y una línea "Pendiente: ..." si algo falta, para poder usarlo
en un script de comprobación o en la costumbre de correrlo antes de entregar.

Uso
---
    ./.venv/Scripts/python.exe verificar_entorno.py
"""

from __future__ import annotations

import os
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

PAQUETES = [
    "langchain-core",
    "langchain-chroma",
    "langchain-google-genai",
    "langchain-deepseek",
    "chromadb",
    "langgraph",
    "python-dotenv",
]

VARIABLES = ["GEMINI_API_KEY", "DEEPSEEK_API_KEY"]

INDICE_DIR = Path("vectorstore")
CORPUS_DIR = Path("corpus")


def main() -> int:
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass

    problemas: list[str] = []
    print("Python:", sys.version.split()[0])
    if sys.version_info < (3, 10):
        problemas.append("usar Python 3.10 o superior")

    for paquete in PAQUETES:
        try:
            print(f"{paquete}: {version(paquete)}")
        except PackageNotFoundError:
            print(f"{paquete}: NO instalado")
            problemas.append(f"instalar {paquete}")

    for nombre in VARIABLES:
        estado = "definida" if os.getenv(nombre) else "FALTA"
        print(f"{nombre}: {estado}")
        if estado == "FALTA":
            problemas.append(f"definir {nombre} en el archivo .env")

    if CORPUS_DIR.is_dir() and any(CORPUS_DIR.glob("*.md")):
        print(f"corpus en {CORPUS_DIR}/: {len(list(CORPUS_DIR.glob('*.md')))} documentos")
    else:
        print(f"corpus en {CORPUS_DIR}/: NO existe o está vacío")
        problemas.append("recuperar el corpus")

    if INDICE_DIR.is_dir() and any(INDICE_DIR.iterdir()):
        print(f"índice en {INDICE_DIR}/: ok")
    else:
        print(f"índice en {INDICE_DIR}/: NO existe o está vacío")
        problemas.append("indexar el corpus con python indexar_corpus.py")

    print()
    if problemas:
        print("Pendiente: " + "; ".join(problemas))
        return 1

    print("Entorno listo para la corrida final")
    return 0


if __name__ == "__main__":
    sys.exit(main())
