"""Evaluación de la recuperación: precisión y recall (Bloque 27).

Qué mide
--------
La recuperación, no la generación. Para cada pregunta se compara el conjunto de
ids que el retriever devolvió contra el conjunto de ids que **debería** devolver,
etiquetado a mano:

- **precisión@k**: de los fragmentos recuperados, qué fracción era relevante.
- **recall@k**: de los fragmentos relevantes, qué fracción se recuperó.

Un recall alto con respuestas malas es un problema de generación; un recall bajo
es un problema de recuperación. Son diagnósticos distintos y por eso se miden
por separado.

Cómo se etiqueta (y la trampa)
-------------------------------
Cada pregunta lleva los ids que la responden, decididos **leyendo el corpus**,
antes de ver qué devuelve el retriever. Etiquetar después de ver el resultado es
lo que convierte una medición en un maquillaje: se ajusta la etiqueta hasta que
el número queda bonito, y el número deja de significar nada.

La mayoría de los casos tiene **un** fragmento que responde, y eso no es
pereza: con un solo id relevante, el recall no puede ser menor que la precisión,
y esa relación sirve como control de que las fórmulas y el `k` están bien
usados. Los dos casos sin fragmento relevante son preguntas que el corpus no
puede responder: ahí el recall no aplica y se excluye del promedio.

Uso
---
    ./.venv/Scripts/python.exe evaluacion_recuperacion.py
    ./.venv/Scripts/python.exe evaluacion_recuperacion.py --k 5 --salida resultados_evaluacion_k5.txt
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import date
from pathlib import Path

from dotenv import load_dotenv
from langchain_chroma import Chroma

import indexar_corpus as ic
from recuperacion import K_POR_DEFECTO

logger = logging.getLogger(__name__)

ARCHIVO_SALIDA = Path("resultados_evaluacion.txt")

# Etiquetas decididas leyendo los 31 fragmentos del corpus (ver
# experimentos/listar_fragmentos.py), no el resultado del retriever.
CASOS_EVALUACION: list[dict] = [
    {
        "pregunta": "¿Qué hace el cliente cuando el proveedor responde 401?",
        "ids_relevantes": {"01_cliente_modular.md#1"},
    },
    {
        "pregunta": "¿Cómo crece la espera entre reintentos y cuál es su techo?",
        "ids_relevantes": {"01_cliente_modular.md#3"},
    },
    {
        "pregunta": "¿Por qué un 429 obliga a bajar el ritmo y un 5xx no?",
        "ids_relevantes": {"01_cliente_modular.md#4"},
    },
    {
        "pregunta": "¿Por qué el costo no se calcula con un solo precio por token?",
        "ids_relevantes": {"02_costos.md#0"},
    },
    {
        "pregunta": "¿Cuánto cuesta un millón de tokens de salida en valle y en punta?",
        "ids_relevantes": {"02_costos.md#2"},
    },
    {
        "pregunta": "¿En qué franja horaria aplica la tarifa de hora punta?",
        "ids_relevantes": {"02_costos.md#3"},
    },
    {
        "pregunta": "¿Qué dimensiones clasifica el prompt de clasificación?",
        "ids_relevantes": {"03_banco_prompts.md#1"},
    },
    {
        "pregunta": "¿Qué niveles de prioridad llevan las recomendaciones?",
        "ids_relevantes": {"03_banco_prompts.md#3"},
    },
    {
        "pregunta": "¿Qué herramienta escribe en disco y qué exige antes de escribir?",
        "ids_relevantes": {"04_agente_herramientas.md#3"},
    },
    {
        "pregunta": "¿Cómo recuerda el agente lo que se dijo en el turno anterior?",
        "ids_relevantes": {"04_agente_herramientas.md#4"},
    },
    {
        "pregunta": "¿Qué se hace para que un test de reintentos no tarde veinte segundos?",
        "ids_relevantes": {"05_pruebas.md#1"},
    },
    {
        "pregunta": "¿Qué pasa con la puntuación al limpiar el texto?",
        "ids_relevantes": {"06_busqueda_pipeline.md#2"},
    },
    {
        "pregunta": "¿Por qué buscar 'demora' no encuentra un párrafo que dice 'latencia'?",
        "ids_relevantes": {"06_busqueda_pipeline.md#0"},
    },
    {
        # Sin ids relevantes: el corpus no habla de licencias. Sirve para
        # comprobar que el sistema no fuerza una respuesta donde no la hay.
        "pregunta": "¿Bajo qué licencia se distribuye el proyecto?",
        "ids_relevantes": set(),
    },
    {
        "pregunta": "¿Cuántos alumnos hay inscritos en el curso de capacitación?",
        "ids_relevantes": set(),
    },
]


def precision_at_k(ids_recuperados: list[str], ids_relevantes: set[str]) -> float:
    """Fracción de los ids recuperados que era relevante.

    Sin nada recuperado la precisión es 0.0: no se puede premiar una búsqueda
    vacía, y dejarla como ``None`` la escondería del promedio.
    """

    if not ids_recuperados:
        return 0.0

    acertados = [identificador for identificador in ids_recuperados if identificador in ids_relevantes]
    return len(acertados) / len(ids_recuperados)


def recall_at_k(ids_recuperados: list[str], ids_relevantes: set[str]) -> float | None:
    """Fracción de los ids relevantes que se recuperó.

    Devuelve ``None`` si el caso no tiene ids relevantes: no hay nada que
    recuperar y contarlo como 0.0 castigaría al retriever por una pregunta que
    el corpus no puede responder.
    """

    if not ids_relevantes:
        return None

    acertados = [identificador for identificador in ids_recuperados if identificador in ids_relevantes]
    return len(acertados) / len(ids_relevantes)


def promediar(valores) -> float:
    """Promedio que ignora los ``None`` y evita dividir por cero."""

    validos = [valor for valor in valores if valor is not None]
    return sum(validos) / len(validos) if validos else 0.0


def evaluar_recuperacion(
    vectorstore: Chroma,
    casos: list[dict] | None = None,
    k: int = K_POR_DEFECTO,
) -> tuple[list[dict], float, float]:
    """Corre cada caso contra el retriever real y promedia precisión y recall.

    Returns:
        Los resultados por pregunta, la precisión promedio y el recall promedio.
    """

    casos = casos if casos is not None else CASOS_EVALUACION
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    resultados: list[dict] = []

    for caso in casos:
        fragmentos = retriever.invoke(caso["pregunta"])
        ids_recuperados = [fragmento.id for fragmento in fragmentos if fragmento.id][:k]
        relevantes = set(caso["ids_relevantes"])

        resultados.append(
            {
                "pregunta": caso["pregunta"],
                "ids_recuperados": ids_recuperados,
                "ids_relevantes": sorted(relevantes),
                "precision": precision_at_k(ids_recuperados, relevantes),
                "recall": recall_at_k(ids_recuperados, relevantes),
            }
        )

    precision_promedio = promediar(r["precision"] for r in resultados)
    recall_promedio = promediar(r["recall"] for r in resultados)
    logger.info(
        "Evaluados %d casos: precisión %.3f, recall %.3f",
        len(resultados),
        precision_promedio,
        recall_promedio,
    )
    return resultados, precision_promedio, recall_promedio


def formatear_resultados(
    resultados: list[dict],
    precision_promedio: float,
    recall_promedio: float,
    configuracion: str,
) -> str:
    """Arma el texto del archivo de resultados, listo para comparar corridas."""

    lineas = [f"Fecha: {date.today()}", f"Configuracion: {configuracion}", ""]

    for resultado in resultados:
        recall = resultado["recall"]
        recall_texto = "n/a" if recall is None else f"{recall:.2f}"
        lineas.append(f"{resultado['precision']:.2f}  {recall_texto}  {resultado['pregunta']}")

    lineas.append("")
    lineas.append(
        f"promedio -> precision: {precision_promedio:.3f}  recall: {recall_promedio:.3f}"
    )
    return "\n".join(lineas) + "\n"


def escribir_resultados(
    ruta: Path,
    resultados: list[dict],
    precision_promedio: float,
    recall_promedio: float,
    configuracion: str,
) -> None:
    """Guarda la corrida sin sobrescribir la anterior.

    Sobrescribir el archivo deja la corrida nueva y borra la referencia contra
    la que se quería comparar, que es justo lo que le da valor a la medición.
    """

    if ruta.exists():
        raise FileExistsError(
            f"{ruta} ya existe. Usa otro nombre (por ejemplo con la fecha) en vez "
            "de sobrescribir una corrida anterior."
        )

    ruta.write_text(
        formatear_resultados(resultados, precision_promedio, recall_promedio, configuracion),
        encoding="utf-8",
    )


def leer_configuracion(vectorstore: Chroma, k: int) -> str:
    """Lee chunk_size y chunk_overlap de la metadata de la colección.

    El material pide escribir esos valores a mano en el archivo de resultados.
    Leerlos del índice evita el error silencioso de comparar dos corridas con la
    configuración mal anotada.
    """

    metadatos = dict(vectorstore._collection.metadata or {})
    tamano = metadatos.get("tamano_chunk", "desconocido")
    solapamiento = metadatos.get("solapamiento", "desconocido")
    modelo = metadatos.get("modelo_embeddings", "desconocido")
    return f"k={k}, chunk_size={tamano}, chunk_overlap={solapamiento}, embeddings={modelo}"


def main(argumentos: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    load_dotenv()

    parser = argparse.ArgumentParser(description="Evalúa la recuperación del flujo RAG.")
    parser.add_argument("--k", type=int, default=K_POR_DEFECTO, help="fragmentos a recuperar")
    parser.add_argument("--salida", type=Path, default=ARCHIVO_SALIDA, help="archivo de resultados")
    opciones = parser.parse_args(argumentos)

    try:
        embeddings = ic.construir_embeddings()
        vectorstore = ic.abrir_indice(embeddings)
        ic.verificar_indice(vectorstore)
        configuracion = leer_configuracion(vectorstore, opciones.k)
    except ic.ErrorCorpus as error:
        print(f"ERROR: {error}")
        return 1

    resultados, precision_promedio, recall_promedio = evaluar_recuperacion(
        vectorstore, k=opciones.k
    )

    for resultado in resultados:
        recall = resultado["recall"]
        recall_texto = "n/a" if recall is None else f"{recall:.2f}"
        print(f"{resultado['precision']:.2f}  {recall_texto}  {resultado['pregunta']}")

    print(f"\n{configuracion}")
    print(f"promedio -> precision: {precision_promedio:.3f}  recall: {recall_promedio:.3f}")

    try:
        escribir_resultados(
            opciones.salida, resultados, precision_promedio, recall_promedio, configuracion
        )
    except FileExistsError as error:
        print(f"\nAVISO: {error}")
        return 1

    print(f"\nResultados guardados en {opciones.salida}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
