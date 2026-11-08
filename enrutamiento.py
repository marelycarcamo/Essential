"""Enrutamiento condicional y orquestación resistente — Bloques 19 y 20.

Caso: el mismo de ``cadena_tres_pasos`` (documentación de ALESSIA → material
de capacitación). Cambia el flujo, no el tema.

Qué agrega el Bloque 19
-----------------------
La cadena del Bloque 18 siempre escribía el material con la misma plantilla.
Aquí se clasifica primero el **nivel del público** y se elige una de tres
rutas, cada una con su propia pedagogía:

- **inicial**   → analogías cotidianas, sin jerga, ejercicio guiado.
- **intermedio**→ ejemplos de uso, ejercicio de aplicación.
- **avanzado**  → decisiones de diseño y trade-offs, ejercicio de extensión.

La clasificación no es un adorno: es la condición que decide qué plantilla se
ejecuta. Y cada ruta tiene un prompt distinto, no la misma plantilla
repetida tres veces con otro nombre.

Qué agrega el Bloque 20
-----------------------
Que un paso que falla no tumbe el flujo:

1. **Timeout por llamada.** Medimos ~23 s por respuesta con ``gemini-3.8-flash``;
   el ``timeout=10`` del material haría fallar *todas* las llamadas y el
   respaldo se comería el trabajo completo.
2. **``with_fallbacks``.** Si la cadena principal revienta por un fallo del
   proveedor, se reintenta con el modelo de respaldo.
3. **Respuesta degradada.** Si tampoco hay respaldo, se devuelve un mensaje
   honesto en vez de una excepción.

El detalle que importa: los fallbacks se limitan a ``ERRORES_DE_PROVEEDOR``.
Por defecto LangChain atrapa ``Exception``, así que un ``KeyError`` por una
clave mal puesta se confundiría con un proveedor caído. Eso ya nos pasó: el
``DeepSeekWrapper`` no es un ``Runnable`` y la cadena no se podía ni armar; con
el ``except`` ancho, el error quedó invisible y el respaldo era código muerto.

Proyecto: ESSENTIAL — actividades prácticas de los Bloques 19 y 20.
"""

from __future__ import annotations

import logging
import os

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import (
    Runnable,
    RunnableBranch,
    RunnableLambda,
    RunnablePassthrough,
)
from langchain_google_genai import ChatGoogleGenerativeAI

from cadena_tres_pasos import (
    ERRORES_DE_PROVEEDOR,
    NIVELES_VALIDOS,
    NIVEL_POR_DEFECTO,
    TEXTO_DE_EJEMPLO,
    configurar_registro,
    normalizar_opcion,
)
from src.deepseek_wrapper import DeepSeekWrapper

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# Clasificación: ¿a qué nivel apunta el documento?
# --------------------------------------------------------------------------

PLANTILLA_NIVEL = PromptTemplate.from_template(
    "Clasifica el nivel del público al que apunta este documento técnico.\n\n"
    "Responde SOLO con una palabra: inicial, intermedio o avanzado.\n"
    "- inicial: no exige conocimientos previos del tema.\n"
    "- intermedio: exige conocer el tema, no la herramienta.\n"
    "- avanzado: exige conocer la herramienta o el dominio.\n\n"
    "Si el documento no permite decidirlo, responde intermedio.\n\n"
    "Documento:\n"
    "{texto}"
)


def interpretar_nivel(texto: str) -> str:
    """Devuelve uno de ``NIVELES_VALIDOS``.

    El modelo puede contestar «Avanzado» con mayúscula o con tilde; comparar
    esa respuesta cruda contra el catálogo mandaría todo al nivel por defecto
    y las tres rutas quedarían reducidas a una.
    """

    return normalizar_opcion(texto, NIVELES_VALIDOS, NIVEL_POR_DEFECTO)


def construir_paso_nivel(modelo: Runnable) -> Runnable:
    """Devuelve el paso que clasifica el documento en un nivel."""

    return (
        PLANTILLA_NIVEL
        | modelo
        | StrOutputParser()
        | interpretar_nivel
    )


# --------------------------------------------------------------------------
# Las tres rutas: misma tarea, pedagogía distinta
# --------------------------------------------------------------------------

PLANTILLA_INICIAL = PromptTemplate.from_template(
    "Prepara material de capacitación para un público que parte de cero sobre "
    "el siguiente documento técnico.\n\n"
    "Reglas:\n"
    "- Explica cada concepto con una analogía de la vida diaria.\n"
    "- Evita la jerga: si un término técnico es inevitable, defínelo en la "
    "misma frase.\n"
    "- Cierra con un ejercicio guiado, paso a paso.\n"
    "- Máximo doscientas palabras.\n\n"
    "Documento:\n"
    "{texto}"
)

PLANTILLA_INTERMEDIA = PromptTemplate.from_template(
    "Prepara material de capacitación para alguien que ya conoce el tema pero "
    "no la herramienta, a partir de este documento técnico.\n\n"
    "Reglas:\n"
    "- Cada concepto con un ejemplo concreto de uso.\n"
    "- Relaciona lo nuevo con lo que la persona ya sabe del tema.\n"
    "- Cierra con un ejercicio de aplicación, de una sola frase.\n"
    "- Máximo doscientas palabras.\n\n"
    "Documento:\n"
    "{texto}"
)

PLANTILLA_AVANZADA = PromptTemplate.from_template(
    "Prepara material de capacitación para alguien que ya usa la herramienta, "
    "a partir de este documento técnico.\n\n"
    "Reglas:\n"
    "- Cada concepto con la decisión de diseño que hay detrás.\n"
    "- Señala un trade-off real de esa decisión.\n"
    "- Cierra con un ejercicio de extensión, de una sola frase.\n"
    "- Máximo doscientas palabras.\n\n"
    "Documento:\n"
    "{texto}"
)


def construir_rutas(modelo: Runnable) -> dict[str, Runnable]:
    """Devuelve una rutina por nivel, cada una con su propia plantilla."""

    return {
        "inicial": PLANTILLA_INICIAL | modelo | StrOutputParser(),
        "intermedio": PLANTILLA_INTERMEDIA | modelo | StrOutputParser(),
        "avanzado": PLANTILLA_AVANZADA | modelo | StrOutputParser(),
    }


def construir_bifurcacion(rutas: dict[str, Runnable]) -> Runnable:
    """Devuelve la bifurcación: dos condiciones y una ruta por defecto.

    El orden importa. ``RunnableBranch`` evalúa en orden y se queda con la
    primera condición que se cumpla; lo que no coincide con ninguna cae en la
    última cadena, sin condición.
    """

    return RunnableBranch(
        (lambda entrada: entrada["nivel"] == "inicial", rutas["inicial"]),
        (lambda entrada: entrada["nivel"] == "avanzado", rutas["avanzado"]),
        rutas[NIVEL_POR_DEFECTO],
    )


def construir_cadena_con_enrutamiento(
    modelo: Runnable,
    rutas: dict[str, Runnable] | None = None,
) -> Runnable:
    """Clasifica el documento y lo manda a la ruta que corresponde.

    Args:
        modelo: modelo que clasifica y redacta.
        rutas: rutas ya armadas. Se inyectan en las pruebas para no gastar
            llamadas y para verificar a cuál se fue.

    ``RunnablePassthrough.assign`` conserva el ``texto`` de la entrada y le
    suma la clave ``nivel``: cada ruta sigue teniendo el documento original.
    """

    rutas = rutas or construir_rutas(modelo)

    return (
        RunnablePassthrough.assign(nivel=construir_paso_nivel(modelo))
        | construir_bifurcacion(rutas)
    )


# --------------------------------------------------------------------------
# Bloque 20: que un paso que falle no tumbe el flujo
# --------------------------------------------------------------------------

RESPUESTA_GENERICA = (
    "No pudimos generar el material en este momento. El documento quedó "
    "registrado; vuelve a intentarlo en unos minutos."
)


def _respuesta_generica(entrada: dict[str, str]) -> str:
    """Último recurso: se responde algo honesto en lugar de una excepción."""

    logger.warning("Todas las rutas fallaron; se entrega la respuesta genérica.")

    return RESPUESTA_GENERICA


def construir_cadena_segura(
    modelo: Runnable,
    modelo_respaldo: Runnable | None = None,
) -> Runnable:
    """Envuelve la cadena con respaldo y respuesta degradada.

    Los fallbacks se limitan a ``ERRORES_DE_PROVEEDOR``: un error de
    programación no se disfraza de caída del servicio.
    """

    respaldos: list[Runnable] = []

    if modelo_respaldo is not None:
        respaldos.append(construir_cadena_con_enrutamiento(modelo_respaldo))

    respaldos.append(RunnableLambda(_respuesta_generica))

    return construir_cadena_con_enrutamiento(modelo).with_fallbacks(
        respaldos,
        exceptions_to_handle=ERRORES_DE_PROVEEDOR,
    )


# --------------------------------------------------------------------------
# Ejecución directa
# --------------------------------------------------------------------------


def main() -> int:
    """Corre la cadena segura de extremo a extremo con el documento de ejemplo."""

    configurar_registro()
    load_dotenv()

    clave = os.getenv("GEMINI_API_KEY")
    if not clave:
        logger.error(
            "Falta GEMINI_API_KEY en el entorno. Se espera en el .env del "
            "proyecto; sin ella no hay llamada al modelo."
        )
        return 1

    # Timeout por llamada, no por cadena: cada invocación hace dos llamadas
    # (clasificar y redactar) y el presupuesto se consume en cada una.
    timeout = float(os.getenv("LLM_TIMEOUT_SECONDS", "60"))

    modelo = ChatGoogleGenerativeAI(
        model=os.getenv("GEMINI_MODEL", "gemini-3.8-flash"),
        google_api_key=clave,
        temperature=0,
        timeout=timeout,
    )

    respaldo: Runnable | None = None
    try:
        # ``DeepSeekWrapper`` expone ``invoke`` pero no es un ``Runnable`` ni un
        # callable: sin este adaptador, LangChain no lo puede componer con ``|``.
        respaldo = RunnableLambda(DeepSeekWrapper().invoke)
    except RuntimeError as error:
        logger.info("Sin modelo de respaldo (%s).", error)

    cadena = construir_cadena_segura(modelo, respaldo)

    logger.info("Documento de entrada: %s...", TEXTO_DE_EJEMPLO[:70])
    logger.info("Resultado:\n%s", cadena.invoke({"texto": TEXTO_DE_EJEMPLO}))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
