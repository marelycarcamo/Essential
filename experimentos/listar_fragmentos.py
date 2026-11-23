"""Lista los fragmentos del índice con su id (Bloque 27).

Para qué sirve
--------------
La evaluación de la recuperación se etiqueta a mano, y para etiquetar hay que
saber qué id corresponde a qué texto. Este script imprime esa relación leyendo
el índice, **no** el resultado del retriever: la etiqueta se decide mirando el
corpus, nunca mirando lo que el retriever devolvió, porque eso convertiría la
medición en un ajuste hasta que el número quede bien.

Uso
---
    ./.venv/Scripts/python.exe experimentos/listar_fragmentos.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv  # noqa: E402

import indexar_corpus as ic  # noqa: E402

LARGO_VISTA_PREVIA = 110


def main() -> int:
    load_dotenv()
    try:
        embeddings = ic.construir_embeddings()
        vectorstore = ic.abrir_indice(embeddings)
        resumen = ic.verificar_indice(vectorstore)
    except ic.ErrorCorpus as error:
        print(f"ERROR: {error}")
        return 1

    datos = vectorstore.get(include=["documents", "metadatas"])
    print(f"Fragmentos: {resumen['fragmentos']}\n")

    for identificador, contenido in zip(datos["ids"], datos["documents"]):
        vista = " ".join(contenido.split())[:LARGO_VISTA_PREVIA]
        print(f"{identificador:28} {vista}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
