"""Agente con dos herramientas propias y su traza de decisión (Bloque 21).

Qué es esto y en qué se diferencia de la cadena de los bloques anteriores
-----------------------------------------------------------------------
Una **cadena** tiene los pasos escritos: analizar, planificar, redactar, siempre
en ese orden. Un **agente** no: recibe un objetivo y decide en cada vuelta si
responde con lo que sabe o si llama a una herramienta. El orden lo pone el
modelo, no el código.

Esa libertad es también el problema: sin traza, un agente es una caja negra que
a veces acierta. Por eso el ejecutor se construye con
``return_intermediate_steps=True``: cada decisión queda registrada y se puede
auditar qué herramienta eligió, con qué argumentos y qué recibió.

Por qué el modelo entra como parámetro
--------------------------------------
``construir_agente`` recibe el modelo ya construido. Eso permite probar el ciclo
completo —decisión, ejecución de la herramienta, respuesta final— con un modelo
falso que responde al instante y sin red, y deja el gasto de cuota sólo para la
verificación real.

Uso:
    .\\\\.venv\\\\Scripts\\\\python.exe agente_con_herramientas.py
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

RAIZ = Path(__file__).resolve().parent
SRC = RAIZ / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cadena_tres_pasos import configurar_registro  # noqa: E402
from herramientas import buscar_en_documentacion, estimar_costo  # noqa: E402

logger = logging.getLogger(__name__)

MODELO_POR_DEFECTO = "deepseek-chat"
TIEMPO_MAXIMO_POR_DEFECTO = 60.0
REINTENTOS_POR_DEFECTO = 0

# Tope de vueltas del agente. No es un detalle de rendimiento: es el límite de
# autonomía. Sin él, un agente que se equivoca de herramienta puede iterar
# indefinidamente y cada vuelta cuesta tokens. El valor cubre el caso real
# (buscar, estimar, responder) con margen para una corrección.
MAXIMO_ITERACIONES = 6

HERRAMIENTAS = [buscar_en_documentacion, estimar_costo]

SISTEMA = """Eres el asistente técnico de este proyecto, que convierte
documentación en material de capacitación.

Dispones de dos herramientas:
- buscar_en_documentacion: para preguntas sobre decisiones, mediciones o límites
  de este proyecto.
- estimar_costo: para saber cuánto costaría una llamada a un modelo.

Reglas:
1. Si la pregunta es sobre este proyecto, busca antes de responder.
2. Cuando uses la documentación, cita el archivo y la línea de donde sale.
3. Si una herramienta no encuentra nada, dilo. No rellenes el vacío con lo que
   creas recordar.
4. Para conversación general responde directo, sin herramientas."""

PLANTILLA = ChatPromptTemplate.from_messages(
    [
        ("system", SISTEMA),
        ("human", "{input}"),
        # El scratchpad es donde el agente escribe lo que ya intentó. Sin este
        # hueco el modelo no ve sus propias decisiones anteriores y repetiría la
        # misma herramienta en cada vuelta.
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)


def construir_modelo(
    tiempo_maximo: float = TIEMPO_MAXIMO_POR_DEFECTO,
    reintentos: int = REINTENTOS_POR_DEFECTO,
):
    """Construye el modelo de DeepSeek por el camino oficial de LangChain.

    Se usa ``langchain-deepseek`` y no el wrapper de ``requests`` de bloques
    anteriores porque un agente necesita que el modelo **pida** herramientas, y
    eso exige el protocolo de *tool calling*, que el wrapper de texto no habla.

    Args:
        tiempo_maximo: segundos de espera por intento.
        reintentos: reintentos internos del cliente. En cero para que un fallo
            se vea de inmediato en vez de alargarse en una escalera de esperas.

    Returns:
        Un ``ChatDeepSeek`` listo para ``bind_tools``.

    Raises:
        RuntimeError: si falta ``DEEPSEEK_API_KEY``.
    """

    from langchain_deepseek import ChatDeepSeek

    clave = os.getenv("DEEPSEEK_API_KEY")

    if not clave:
        raise RuntimeError(
            "Falta DEEPSEEK_API_KEY en el entorno: no se puede construir el modelo."
        )

    return ChatDeepSeek(
        model=os.getenv("DEEPSEEK_MODEL", MODELO_POR_DEFECTO),
        api_key=clave,
        temperature=0,
        timeout=tiempo_maximo,
        max_retries=reintentos,
    )


def construir_agente(modelo) -> AgentExecutor:
    """Arma el ejecutor del agente con las herramientas del proyecto.

    Args:
        modelo: modelo de chat con soporte de *tool calling*, ya construido.

    Returns:
        El ejecutor, con la traza de decisiones activada.
    """

    agente = create_tool_calling_agent(modelo, HERRAMIENTAS, PLANTILLA)

    return AgentExecutor(
        agent=agente,
        tools=HERRAMIENTAS,
        return_intermediate_steps=True,
        verbose=False,
        max_iterations=MAXIMO_ITERACIONES,
        # Un error de formato del modelo no debe matar el proceso: se le
        # devuelve y se le da otra vuelta.
        handle_parsing_errors=True,
    )


def herramientas_usadas(pasos: list[tuple]) -> list[str]:
    """Nombres de las herramientas que el agente decidió usar, en orden.

    Args:
        pasos: ``intermediate_steps`` del resultado del ejecutor.

    Returns:
        Lista de nombres. Vacía si el agente respondió sin herramientas.
    """

    return [accion.tool for accion, _ in pasos]


def main() -> int:
    """Ejecuta las verificaciones reales del bloque.

    Son invocaciones de verdad contra el proveedor, una por herramienta y una
    que no debería usar ninguna. Se imprimen la traza y la respuesta para poder
    comprobarlas a ojo, que es lo que pide la actividad.
    """

    configurar_registro()
    load_dotenv(RAIZ / ".env")

    try:
        ejecutor = construir_agente(construir_modelo())
    except RuntimeError as error:
        logger.error("%s", error)
        return 1

    pruebas = (
        (
            "debe usar buscar_en_documentacion",
            "¿Por qué se descartó GPT-4o mini como modelo principal?",
        ),
        (
            "debe usar estimar_costo",
            "¿Cuánto cuesta generar un material de 1500 tokens de entrada y "
            "400 de salida con deepseek-flash?",
        ),
        (
            "no necesita herramienta",
            "En una frase, ¿qué puedes hacer?",
        ),
    )

    for expectativa, pregunta in pruebas:
        logger.info("")
        logger.info("=" * 66)
        logger.info("PREGUNTA: %s", pregunta)
        logger.info("ESPERA: %s", expectativa)

        try:
            resultado = ejecutor.invoke({"input": pregunta})
        except Exception as error:  # noqa: BLE001 - la corrida es manual
            logger.error("La invocación falló: %s: %s", type(error).__name__, error)
            continue

        usadas = herramientas_usadas(resultado["intermediate_steps"])

        for accion, observacion in resultado["intermediate_steps"]:
            logger.info(
                "  · herramienta: %s · argumentos: %s",
                accion.tool,
                accion.tool_input,
            )
            logger.info("    devolvió: %s", str(observacion)[:300])

        logger.info(
            "  DECISIÓN: %s",
            ", ".join(usadas) if usadas else "respondió sin herramientas",
        )
        logger.info("  RESPUESTA: %s", resultado["output"])

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
