"""Punto de supervisión humana para desplegar una versión de prompt (Bloque 32).

Qué resuelve
------------
Versionar el prompt permite auditar **después**. Hay acciones que no deberían
ejecutarse solas **antes**, aunque el resto del sistema funcione perfecto:
desplegar a producción una versión nueva de un prompt es una de ellas, porque
cambia el comportamiento de todas las respuestas futuras y no se revierte
mirando atrás.

El grafo de LangGraph pausa la ejecución con ``interrupt()`` y no continúa hasta
que alguien responda con ``Command(resume=...)``. Si el proceso se cae mientras
espera la aprobación, con un checkpointer durable la aprobación pendiente
sobrevive al reinicio; con uno en memoria, no.

Por qué el guardia antes de reanudar
-----------------------------------
Reanudar un hilo que ya se resolvió no lanza una excepción: devuelve el estado
final sin ejecutar nada, o aplica la aprobación a la solicitud equivocada. Por
eso ``resolver`` revisa ``get_state`` y falla con un mensaje claro si no hay
nada pendiente. Un punto de supervisión que aprueba algo que nadie miró es peor
que no tenerlo: da la falsa garantía de que hubo control.

Qué acciones merecen supervisión
--------------------------------
Sí: desplegar un prompt a producción, enviar una comunicación en nombre de la
organización, ejecutar algo con costo económico o irreversible.
No: responder una consulta con información ya recuperada y citada. Pedir
aprobación por cada respuesta individual no escala y no es el problema que esta
técnica resuelve.

Uso
---
    ./.venv/Scripts/python.exe grafo_supervision.py
    ./.venv/Scripts/python.exe grafo_supervision.py --durable
"""

from __future__ import annotations

import argparse
import logging
import sys
import uuid
from pathlib import Path
from typing import TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

logger = logging.getLogger(__name__)

ARCHIVO_CHECKPOINTS = Path("checkpoints.db")
MOTIVO = "aprobar nueva version de prompt en produccion"


class EstadoDespliegue(TypedDict, total=False):
    """Estado del despliegue de un prompt.

    ``total=False`` porque el estado arranca sin ``aprobado`` ni ``aplicado``: los
    pone el grafo. Un nodo que devuelve una clave que no está declarada acá es
    ignorado por LangGraph sin avisar, que es un modo de falla silenciosa.
    """

    huella_nueva: str
    aprobado: bool
    aplicado: bool


def proponer(estado: EstadoDespliegue) -> dict:
    """Nodo previo a la supervisión: deja constancia de qué se quiere desplegar."""

    logger.info(
        "propuesta de despliegue recibida: huella %s", estado.get("huella_nueva")
    )
    return {}


def supervisar(estado: EstadoDespliegue) -> dict:
    """Nodo que pausa el grafo y espera una decisión humana.

    ``interrupt`` devuelve el valor que se pase a ``Command(resume=...)``. Se
    exige un booleano explícito: ``Command(resume="no")`` activaría la versión,
    porque un texto no vacío es verdadero en Python.
    """

    decision = interrupt(
        {
            "motivo": MOTIVO,
            "huella_propuesta": estado.get("huella_nueva"),
        }
    )
    if not isinstance(decision, bool):
        raise ValueError(
            f"La decisión debe ser un booleano (True/False), no {type(decision).__name__}: "
            "un texto como 'no' activaría la versión."
        )
    return {"aprobado": decision}


def aplicar(estado: EstadoDespliegue) -> dict:
    """Nodo posterior: activa o rechaza según la decisión, y lo deja en el estado."""

    aprobado = bool(estado.get("aprobado"))
    huella = estado.get("huella_nueva")
    if aprobado:
        logger.info("version %s activada", huella)
    else:
        logger.warning("version %s rechazada, se mantiene la anterior", huella)
    return {"aplicado": aprobado}


def construir_grafo(durable: bool = False, ruta_checkpoints: Path | str = ARCHIVO_CHECKPOINTS):
    """Compila el grafo de supervisión.

    Args:
        durable: si es ``True`` usa un checkpointer en SQLite, así la aprobación
            pendiente sobrevive a un reinicio del proceso. Con ``False`` usa
            memoria del proceso, que alcanza para demostrar el mecanismo pero no
            para un punto de supervisión que una persona atiende horas después.
        ruta_checkpoints: archivo donde se guardan los checkpoints durables.
    """

    grafo = StateGraph(EstadoDespliegue)
    grafo.add_node("proponer", proponer)
    grafo.add_node("supervisar", supervisar)
    grafo.add_node("aplicar", aplicar)
    grafo.add_edge(START, "proponer")
    grafo.add_edge("proponer", "supervisar")
    grafo.add_edge("supervisar", "aplicar")
    grafo.add_edge("aplicar", END)

    if durable:
        import sqlite3

        from langgraph.checkpoint.sqlite import SqliteSaver

        conexion = sqlite3.connect(str(ruta_checkpoints), check_same_thread=False)
        return grafo.compile(checkpointer=SqliteSaver(conexion))

    return grafo.compile(checkpointer=InMemorySaver())


def nuevo_thread_id(prefijo: str = "despliegue") -> str:
    """Un hilo por solicitud de aprobación: reutilizar uno ya resuelto no re-pausa."""

    return f"{prefijo}-{uuid.uuid4().hex[:8]}"


def solicitar_aprobacion(
    app,
    huella: str,
    thread_id: str | None = None,
) -> tuple[dict, dict]:
    """Invoca el grafo hasta la pausa.

    Returns:
        La configuración del hilo (para reanudarlo después) y el estado pausado.
    """

    config = {"configurable": {"thread_id": thread_id or nuevo_thread_id()}}
    estado = app.invoke({"huella_nueva": huella}, config=config)
    return config, estado


def esta_pendiente(app, config: dict) -> bool:
    """Indica si el hilo tiene una aprobación sin resolver."""

    return bool(app.get_state(config).next)


def resolver(app, config: dict, aprobar: bool) -> dict:
    """Reanuda el hilo con la decisión humana.

    Raises:
        ValueError: si el hilo no tiene nada pendiente (ya se resolvió, el
            ``thread_id`` no existe o el proceso se reinició sin checkpointer
            durable) o si la decisión no es booleana.
    """

    if not isinstance(aprobar, bool):
        raise ValueError("La decisión debe ser True o False, explícita")

    if not esta_pendiente(app, config):
        raise ValueError(
            "este thread_id no tiene ninguna aprobacion pendiente"
        )

    return app.invoke(Command(resume=aprobar), config=config)


def main(argumentos: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Punto de supervisión de despliegues.")
    parser.add_argument(
        "--durable",
        action="store_true",
        help="guarda los checkpoints en SQLite en vez de memoria",
    )
    parser.add_argument(
        "huella",
        nargs="?",
        default="ba2bcb4f4f13",
        help="huella de la versión de prompt que se quiere desplegar",
    )
    opciones = parser.parse_args(argumentos)

    app = construir_grafo(durable=opciones.durable)
    huella = opciones.huella

    print("--- camino de aprobación ---")
    config, pausado = solicitar_aprobacion(app, huella)
    print("1) tras invocar:", pausado)
    print("2) pendiente:", app.get_state(config).interrupts[0].value)
    print("   ¿sigue pendiente?", esta_pendiente(app, config))
    final = resolver(app, config, True)
    print("3) tras aprobar:", final)
    print("   ¿sigue pendiente?", esta_pendiente(app, config))

    print("\n--- camino de rechazo (hilo nuevo) ---")
    config2, _ = solicitar_aprobacion(app, huella)
    rechazo = resolver(app, config2, False)
    print("4) tras rechazar:", rechazo)

    try:
        resolver(app, config2, True)
    except ValueError as error:
        print(f"5) segunda aprobación del mismo hilo: {error}")

    print("\nProceso terminado sin errores.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
