"""Agente con herramientas, traza de decisión y memoria (Bloques 21 y 22).

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

Por qué hay herramientas en ``src/`` y una acá
----------------------------------------------
Buscar en la documentación y estimar un costo son **capacidades del dominio**:
no saben que existe un agente y servirían igual en un servicio HTTP. Viven en
``src/herramientas.py``.

Guardar un material es distinto: lo que la vuelve segura no es lo que hace, sino
la **política** que la rodea —no escribe sin aprobación humana y no escribe
material que no cumpla el contrato—. Esa política es del agente, y por eso vive
aquí, al lado del tope de vueltas y del prompt.

Por qué el modelo entra como parámetro
--------------------------------------
``construir_agente`` y ``construir_agente_con_memoria`` reciben el modelo ya
construido. Eso permite probar el ciclo completo —decisión, ejecución de la
herramienta, memoria entre turnos— con un modelo falso que responde al instante
y sin red, y deja el gasto de cuota sólo para la verificación real.

Uso:
    .\\\\.venv\\\\Scripts\\\\python.exe agente_con_herramientas.py
"""

from __future__ import annotations

import logging
import os
import re
import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.tools import tool
from pydantic import BaseModel, Field

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

# Dónde deja sus materiales la herramienta de escritura. Es salida del agente,
# no fuente del proyecto: va en .gitignore, como los logs.
DIRECTORIO_MATERIAL = RAIZ / "material_generado"
SUFIJO_MATERIAL = ".md"

# Un nombre de archivo que llega desde el modelo no es de fiar: si trae "/",
# "\" o "..", la escritura deja de ser "un material más" y pasa a ser una
# escritura arbitraria en el disco.
PATRON_NOMBRE_SEGURO = re.compile(r"^[a-z0-9][a-z0-9._-]{0,80}$")

SISTEMA = """Eres el asistente técnico de este proyecto, que convierte
documentación en material de capacitación.

Dispones de tres herramientas:
- buscar_en_documentacion: para preguntas sobre decisiones, mediciones o límites
  de este proyecto.
- estimar_costo: para saber cuánto costaría una llamada a un modelo.
- guardar_material: para dejar un material aprobado en disco. Escribe, así que
  exige aprobación humana explícita.

Reglas:
1. Si la pregunta es sobre este proyecto, busca antes de responder.
2. Cuando uses la documentación, cita el archivo y la línea de donde sale.
3. Si una herramienta no encuentra nada, dilo. No rellenes el vacío con lo que
   creas recordar.
4. No guardes nada sin que la persona lo haya revisado y aprobado.
5. Para conversación general responde directo, sin herramientas."""


class ArgumentosGuardarMaterial(BaseModel):
    """Argumentos de la herramienta de escritura, descritos campo por campo.

    El docstring de una herramienta describe la herramienta completa; con más de
    un argumento, el modelo necesita saber qué representa cada uno. Pydantic
    además valida los tipos antes de que la función llegue a ejecutarse.
    """

    nombre_archivo: str = Field(
        description=(
            "Nombre del archivo a crear, sin carpetas, en minúsculas y con "
            "extensión .md. Por ejemplo: 'material-inicial-latencia.md'."
        )
    )
    contenido: str = Field(
        description=(
            "Material completo en markdown. Debe traer las secciones "
            "'Objetivo:', 'Conceptos:' y 'Ejercicio:', y citar el documento "
            "con líneas que empiezan por '>'."
        )
    )
    aprobacion_humana: bool = Field(
        default=False,
        description=(
            "True solo si una persona leyó el material y autorizó guardarlo. "
            "Con False la herramienta no escribe nada y devuelve el motivo."
        ),
    )


@tool("guardar_material", args_schema=ArgumentosGuardarMaterial)
def guardar_material(
    nombre_archivo: str,
    contenido: str,
    aprobacion_humana: bool = False,
) -> str:
    """Guarda un material de capacitación aprobado en la carpeta de salida.

    Úsala solo cuando la persona haya leído el material y confirmado que se
    guarde. Es la única herramienta que escribe: sin aprobación explícita no
    toca el disco.

    Args:
        nombre_archivo: nombre del archivo, sin carpetas y con extensión .md.
        contenido: material completo en markdown.
        aprobacion_humana: True solo si la persona ya lo revisó y lo aprobó.

    Returns:
        La ruta donde quedó guardado, o el motivo por el que no se guardó.
    """

    if not aprobacion_humana:
        return (
            "No guardé nada: falta la aprobación humana. Muéstrale el material "
            "a la persona, pídele que lo revise y, cuando confirme, vuelve a "
            "llamarme con aprobacion_humana=True."
        )

    nombre = nombre_archivo.strip().lower()
    if not nombre.endswith(SUFIJO_MATERIAL):
        nombre += SUFIJO_MATERIAL

    if not PATRON_NOMBRE_SEGURO.match(nombre):
        return (
            f"El nombre «{nombre_archivo}» no sirve: usa minúsculas, números, "
            "guiones o guiones bajos, sin carpetas ni rutas."
        )

    if not contenido.strip():
        return "El material está vacío: no hay nada que guardar."

    # Importación diferida: el verificador vive en el script de enrutamiento, y
    # traerlo en el arranque cargaría el SDK de Google para todos los usos del
    # agente, incluso cuando no se guarda nada.
    from enrutamiento import verificar_material

    revision = verificar_material(contenido, _corpus_de_documentacion())

    if not revision.aprueba:
        return (
            f"No guardé el material: no cumple el contrato ({revision.resumen()}). "
            "Corrígelo y vuelve a intentarlo."
        )

    destino = DIRECTORIO_MATERIAL / nombre

    try:
        destino.parent.mkdir(parents=True, exist_ok=True)
        destino.write_text(contenido, encoding="utf-8")
    except OSError as error:
        logger.error("No se pudo escribir %s: %s", destino, error)
        return f"No pude escribir el archivo: {error}"

    # La ruta se muestra relativa si se puede: una ruta absoluta con el usuario
    # de la máquina no aporta nada en un log. Si la carpeta de salida está
    # fuera del proyecto, se muestra tal cual en vez de fallar.
    try:
        mostrada = destino.relative_to(RAIZ)
    except ValueError:
        mostrada = destino

    return (
        f"Guardado en {mostrada} "
        f"({revision.palabras} palabras propias, contrato cumplido)."
    )


def _corpus_de_documentacion() -> str:
    """Todo el texto de la documentación, junto, para verificar citas.

    Verificar una cita contra la documentación completa es más permisivo que
    contra el documento exacto del que salió, pero no requiere que el modelo
    arrastre el documento entero en la llamada: detecta la cita inventada, que
    es lo que se busca, sin pedirle al agente que copie su fuente.
    """

    from herramientas import cargar_fragmentos

    return " ".join(fragmento.texto for fragmento in cargar_fragmentos())


HERRAMIENTAS = [buscar_en_documentacion, estimar_costo, guardar_material]

PLANTILLA = ChatPromptTemplate.from_messages(
    [
        ("system", SISTEMA),
        # El historial va antes del mensaje humano: es lo que ya se dijo.
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        # El scratchpad es donde el agente escribe lo que ya intentó en esta
        # vuelta. Sin este hueco no vería sus propias decisiones anteriores y
        # repetiría la misma herramienta en cada iteración.
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


def construir_agente_con_memoria(
    modelo,
    historiales: dict[str, InMemoryChatMessageHistory] | None = None,
) -> RunnableWithMessageHistory:
    """Envuelve el ejecutor para que recuerde lo dicho en cada sesión.

    La memoria no vive dentro del agente: vive en un diccionario indexado por
    ``session_id``. El envoltorio se encarga de leer el historial antes de cada
    llamada y de guardar el turno después.

    Args:
        modelo: modelo de chat con soporte de *tool calling*, ya construido.
        historiales: diccionario donde se acumulan las sesiones. Se pasa desde
            afuera para poder inspeccionarlo o aislarlo en las pruebas.

    Returns:
        El agente con memoria. Cada ``invoke`` necesita un ``config`` con
        ``session_id``; para eso está ``configurar_sesion``.
    """

    ejecutor = construir_agente(modelo)
    sesiones = historiales if historiales is not None else {}

    def obtener_historial(session_id: str) -> InMemoryChatMessageHistory:
        """Devuelve el historial de esa sesión, creándolo la primera vez."""

        if session_id not in sesiones:
            sesiones[session_id] = InMemoryChatMessageHistory()

        return sesiones[session_id]

    return RunnableWithMessageHistory(
        ejecutor,
        obtener_historial,
        input_messages_key="input",
        history_messages_key="chat_history",
    )


def configurar_sesion(session_id: str) -> dict:
    """Arma el ``config`` que espera el agente con memoria.

    Args:
        session_id: identificador de la conversación. Dos personas distintas
            nunca deben compartirlo: es lo único que separa sus historiales.

    Returns:
        El diccionario que se pasa como segundo argumento de ``invoke``.
    """

    return {"configurable": {"session_id": session_id}}


def herramientas_usadas(pasos: list[tuple]) -> list[str]:
    """Nombres de las herramientas que el agente decidió usar, en orden.

    Args:
        pasos: ``intermediate_steps`` del resultado del ejecutor.

    Returns:
        Lista de nombres. Vacía si el agente respondió sin herramientas.
    """

    return [accion.tool for accion, _ in pasos]


def ejecutar_turno(agente, pregunta: str, config: dict, etiqueta: str) -> dict | None:
    """Hace un turno, lo deja en el log y devuelve el resultado.

    Args:
        agente: agente con memoria ya construido.
        pregunta: texto de la persona.
        config: ``config`` con el ``session_id`` de la conversación.
        etiqueta: encabezado legible para la salida.

    Returns:
        El resultado del turno, o ``None`` si la llamada falló.
    """

    logger.info("")
    logger.info("=" * 66)
    logger.info("%s", etiqueta)
    logger.info("PREGUNTA: %s", pregunta)

    try:
        resultado = agente.invoke({"input": pregunta}, config)
    except Exception as error:  # noqa: BLE001 - la corrida es manual
        logger.error("La invocación falló: %s: %s", type(error).__name__, error)
        return None

    for accion, observacion in resultado["intermediate_steps"]:
        logger.info("  · herramienta: %s · argumentos: %s", accion.tool, accion.tool_input)
        logger.info("    devolvió: %s", str(observacion)[:240])

    usadas = herramientas_usadas(resultado["intermediate_steps"])
    logger.info("  DECISIÓN: %s", ", ".join(usadas) or "respondió sin herramientas")
    logger.info("  RESPUESTA: %s", resultado["output"])

    return resultado


def main() -> int:
    """Verifica la memoria: dos turnos que se recuerdan y dos sesiones que no.

    Es la comprobación que pide la actividad. La primera parte demuestra que la
    segunda pregunta usa información de la primera; la segunda parte demuestra
    que cambiar el ``session_id`` corta esa continuidad, que es lo que impide
    filtrar una conversación a otra persona.
    """

    configurar_registro()
    load_dotenv(RAIZ / ".env")

    historiales: dict[str, InMemoryChatMessageHistory] = {}

    try:
        agente = construir_agente_con_memoria(construir_modelo(), historiales)
    except RuntimeError as error:
        logger.error("%s", error)
        return 1

    sesion_uno = configurar_sesion("revision-material")
    sesion_dos = configurar_sesion("otra-persona")

    ejecutar_turno(
        agente,
        "¿Por qué se descartó GPT-4o mini como modelo principal?",
        sesion_uno,
        "TURNO 1 · sesión «revision-material»",
    )

    ejecutar_turno(
        agente,
        "¿Y cuánto costaría rehacer ese análisis con 3000 tokens de entrada y "
        "800 de salida?",
        sesion_uno,
        "TURNO 2 · misma sesión · «ese análisis» solo se entiende con memoria",
    )

    ejecutar_turno(
        agente,
        "¿De qué modelo te acabo de preguntar?",
        sesion_dos,
        "SESIÓN DISTINTA · no debería saberlo",
    )

    logger.info("")
    logger.info("=" * 66)
    for session_id, historial in historiales.items():
        logger.info(
            "sesión «%s»: %d mensajes en su historial",
            session_id,
            len(historial.messages),
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
