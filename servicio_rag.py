"""Servicio RAG integrado: indexar, recuperar, filtrar, citar y registrar (Gate 2).

Qué reúne
---------
Es el punto donde las piezas del Módulo 2 corren juntas, que es lo que el Gate
evalúa y lo que ninguna prueba individual garantiza:

1. **Índice persistido** en disco, sincronizado de forma incremental: si el corpus
   no cambió, la corrida no gasta una sola llamada de embedding.
2. **Recuperación semántica** con `k=3`, justificado con precisión y recall
   medidos (Bloque 27).
3. **Filtro de inyección** sobre lo recuperado, antes de armar el contexto
   (Bloque 31).
4. **Citación de fuentes**: la respuesta usa solo el contexto y numera sus citas
   (Bloques 26 y 32).
5. **Logging estructurado con `trace_id`**, sin contenido crudo, y **registro de
   consumo** con la huella del prompt que generó cada respuesta (Bloques 31 y 32).
6. **Punto de supervisión humana** para desplegar una versión de prompt
   (Bloque 32), disponible con `--supervision`.

Por qué el servicio recibe el índice y el modelo como parámetros
----------------------------------------------------------------
Para poder probar el flujo completo sin red ni cuota: las pruebas inyectan
embeddings deterministas y un modelo doble, y así los criterios no compensables se
verifican en cada corrida de la suite. La corrida real los construye desde el
entorno. La estructura del pipeline es la misma en los dos casos; si no lo fuera,
la prueba no probaría nada.

Uso
---
    python servicio_rag.py                                  # consulta real
    python servicio_rag.py --pregunta "¿qué hace el cliente ante un 401?"
    python servicio_rag.py --supervision                    # + grafo de aprobación
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document

import indexar_corpus as ic
from grafo_supervision import construir_grafo, solicitar_aprobacion, resolver
from logging_rag import (
    configurar_logging,
    evento,
    huella,
    nuevo_trace_id,
    responder_con_documentos,
    trace_id_var,
)
from recuperacion import K_POR_DEFECTO, construir_modelo
from registro_prompts import PROMPT_ACTIVO, huella_prompt

logger = logging.getLogger(__name__)

PREGUNTA_POR_DEFECTO = "¿Qué hace el cliente cuando el proveedor responde 401?"


@dataclass
class ServicioRAG:
    """Servicio con un índice y un modelo ya resueltos."""

    vectorstore: Chroma
    modelo: object
    modelo_cobrado: str = "deepseek-flash"
    k: int = K_POR_DEFECTO
    huella_prompt: str | None = None

    def recuperar(self, pregunta: str) -> list[Document]:
        """Trae los ``k`` fragmentos más parecidos por significado."""

        retriever = self.vectorstore.as_retriever(search_kwargs={"k": self.k})
        return retriever.invoke(pregunta)

    def responder(self, pregunta: str) -> dict:
        """Corre una consulta completa y devuelve la respuesta con su traza.

        La traza empieza acá, antes de recuperar, para que el evento de
        recuperación quede dentro del mismo ``trace_id`` que la generación.
        """

        token = trace_id_var.set(nuevo_trace_id())
        auditoria: list[dict] = []

        try:
            evento(
                auditoria,
                "consulta_recibida",
                pregunta_len=len(pregunta),
                pregunta_huella=huella(pregunta),
                k=self.k,
            )

            documentos = self.recuperar(pregunta)
            evento(auditoria, "documentos_recuperados", n=len(documentos), k=self.k)

            return responder_con_documentos(
                pregunta, documentos, self.modelo, auditoria, self.huella_prompt
            )
        finally:
            trace_id_var.reset(token)


def construir_servicio(
    embeddings=None,
    modelo=None,
    k: int = K_POR_DEFECTO,
    directorio: Path = ic.DIRECTORIO_VECTORSTORE,
    corpus: Path = ic.DIRECTORIO_CORPUS,
) -> tuple[ServicioRAG, dict]:
    """Sincroniza el índice con el corpus y devuelve el servicio listo.

    La sincronización es incremental: un corpus sin cambios no embebe nada. Eso es
    lo que hace que el índice sea persistido y no reconstruido en cada corrida.

    Returns:
        El servicio y el resumen de la sincronización.
    """

    embeddings = embeddings or ic.construir_embeddings()
    documentos = ic.cargar_documentos(corpus)
    sincronizacion = ic.actualizar_indice(embeddings, documentos, directorio=directorio)
    vectorstore = ic.abrir_indice(embeddings, directorio)
    ic.verificar_indice(vectorstore)

    servicio = ServicioRAG(
        vectorstore=vectorstore,
        modelo=modelo or construir_modelo(),
        modelo_cobrado=os.environ.get("DEEPSEEK_MODEL", "deepseek-flash"),
        k=k,
        huella_prompt=huella_prompt(PROMPT_ACTIVO),
    )
    return servicio, sincronizacion


def main(argumentos: list[str] | None = None) -> int:
    configurar_logging()
    load_dotenv()

    parser = argparse.ArgumentParser(description="Servicio RAG del proyecto.")
    parser.add_argument("--pregunta", default=PREGUNTA_POR_DEFECTO)
    parser.add_argument("--k", type=int, default=K_POR_DEFECTO)
    parser.add_argument(
        "--supervision",
        action="store_true",
        help="además corre el punto de supervisión humana de un despliegue de prompt",
    )
    opciones = parser.parse_args(argumentos)

    try:
        servicio, sincronizacion = construir_servicio(k=opciones.k)
    except ic.ErrorCorpus as error:
        print(f"ERROR: {error}")
        return 1

    print("--- índice ---")
    print(json.dumps(sincronizacion, indent=2, ensure_ascii=False))
    print()

    resultado = servicio.responder(opciones.pregunta)

    print("--- respuesta del servicio ---")
    print(json.dumps(resultado, indent=2, ensure_ascii=False))

    if not resultado["fuentes"]:
        logger.warning("la respuesta no cita ninguna fuente")

    if opciones.supervision:
        print("\n--- punto de supervisión humana ---")
        app = construir_grafo()
        config, pausado = solicitar_aprobacion(app, huella_prompt(PROMPT_ACTIVO))
        print("pausado:", app.get_state(config).interrupts[0].value)
        final = resolver(app, config, True)
        print("tras aprobar:", final)

    print("\nProceso terminado sin errores.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
