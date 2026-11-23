"""Verificación del índice persistido (Bloque 24).

Qué comprueba
-------------
1. Que el índice exista en disco y tenga el número de fragmentos esperado,
   reconectándose desde un proceso nuevo: si el script de indexación hubiera
   dejado todo en memoria, aquí habría cero.
2. Que la metadata de la colección declare el modelo de embeddings con el que se
   construyó.
3. Que una búsqueda por significado traiga fragmentos del documento correcto,
   usando palabras distintas a las del corpus. Eso es lo que la búsqueda por
   palabra clave de ``src/herramientas.py`` no puede hacer.

Costo: una llamada de embedding por consulta.

Uso
---
    ./.venv/Scripts/python.exe experimentos/verificar_indice.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv  # noqa: E402

import indexar_corpus as ic  # noqa: E402

CONSULTAS = [
    # No usa la palabra "reintento" ni "backoff": prueba el salto semántico.
    "¿Qué pasa si el proveedor se cae y la llamada falla varias veces?",
    # No usa "costo" ni "tarifa".
    "¿Cuánto cuesta cada llamada al modelo?",
    # No usa "prueba" ni "pytest".
    "¿Cómo se comprueba que el código funciona antes de publicarlo?",
]


def main() -> int:
    load_dotenv()
    try:
        embeddings = ic.construir_embeddings()
        vectorstore = ic.abrir_indice(embeddings)
        resumen = ic.verificar_indice(vectorstore)
    except ic.ErrorCorpus as error:
        print(f"ERROR: {error}")
        return 1

    print(f"Fragmentos en disco: {resumen['fragmentos']}")
    print(f"Metadata de la colección: {resumen['metadatos']}\n")

    for consulta in CONSULTAS:
        resultados = vectorstore.similarity_search_with_score(consulta, k=3)
        print(f"Pregunta: {consulta}")
        for documento, distancia in resultados:
            fuente = documento.metadata.get("fuente", "desconocida")
            primeros = documento.page_content.strip().splitlines()[0][:70]
            print(f"  [{distancia:.3f}] {fuente}: {primeros}")
        print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
