"""Enrutamiento condicional y orquestación resistente — Bloques 19 y 20.

Caso: el mismo de ``cadena_tres_pasos`` (documentación de ALESSIA → material
de capacitación). Cambia el flujo, no el tema.

Qué agrega el Bloque 19
-----------------------
La cadena del Bloque 18 siempre escribía el material con la misma plantilla.
Aquí se elige una de tres rutas según el **nivel del público**, cada una con
su propia pedagogía:

- **inicial**   → analogías cotidianas, sin jerga, ejercicio guiado.
- **intermedio**→ ejemplos de uso, ejercicio de aplicación.
- **avanzado**  → decisiones de diseño y trade-offs, ejercicio de extensión.

Cada ruta tiene un prompt distinto, no la misma plantilla repetida tres veces
con otro nombre.

Qué agrega el Bloque 20
-----------------------
Que un paso que falla no tumbe el flujo:

1. **Timeout acotado de verdad.** En ``ChatGoogleGenerativeAI`` el ``timeout``
   es *por intento*, y la librería trae ``max_retries=6``: el techo real es
   ``timeout × 7``. Medido: una llamada a la ruta avanzada terminó en **787 s**
   con ``timeout=240`` y ``504 DEADLINE_EXCEEDED``. Por eso el modelo se arma
   con ``max_retries`` explícito: un techo que no se puede calcular no es un
   techo, y el ``timeout=10`` del material habría dejado el fallo a merced de
   los reintentos.
2. **``with_fallbacks``.** Si la cadena principal revienta por un fallo del
   proveedor, se reintenta con el modelo de respaldo. El principal es
   **DeepSeek** y el respaldo **Gemini**, no al revés: el material del curso
   usa Gemini, pero se cae seguido y la medición lo confirma — la ruta
   avanzada devuelve 504 sin completar, mientras DeepSeek la resuelve en
   21,5 s. Es el mismo orden que fija ``docs/decision-arquitectura.md``.
3. **Respuesta degradada.** Si tampoco hay respaldo, se devuelve un mensaje
   honesto en vez de una excepción.

El detalle que importa: los fallbacks se limitan a ``ERRORES_DE_PROVEEDOR``.
Por defecto LangChain atrapa ``Exception``, así que un ``KeyError`` por una
clave mal puesta se confundiría con un proveedor caído. Eso ya nos pasó: el
``DeepSeekWrapper`` no es un ``Runnable`` y la cadena no se podía ni armar; con
el ``except`` ancho, el error quedó invisible y el respaldo era código muerto.

Qué agrega esta revisión (requisitos del Gate 2, no del material)
-----------------------------------------------------------------
El Gate 2 pide un servicio con orquestación, métricas de evaluación y
documentación. Eso cambia tres decisiones que el ejemplo del curso no toma:

1. **El nivel es un parámetro, no una adivinanza.** Un servicio tiene que ser
   determinista: si el modelo clasifica, el mismo documento cae en rutas
   distintas según el día y las métricas se mueven sin que nadie toque el
   código. Además cada clasificación es una llamada que se paga en cada
   request. La clasificación por modelo queda como **respaldo** para cuando el
   campo no viene, y se registra en el log.
2. **Formato fijo y verificable.** Con secciones obligatorias se puede medir
   algo distinto de la similitud con una referencia escrita a mano — que es la
   métrica que el Bloque 17 demostró que mide extensión, no calidad.
3. **Citas literales.** Cada concepto cita una frase textual del documento. La
   cita se verifica por substring, sin gastar una llamada: es el primer
   eslabón de la fidelidad que el Gate 2 exige y que después se conecta con el
   retrieval.

Limitación declarada: ``verificar_material`` **informa**, no bloquea. Un
material que no cumple el formato se registra como advertencia y se devuelve
igual. Convertirlo en un portón (regenerar o rechazar) es una decisión de
producto que todavía no está tomada.

Dos fallos que aparecieron al correr de verdad
----------------------------------------------
La primera corrida real con el nivel como parámetro murió, y no por el modelo:

1. **``with_fallbacks`` tapaba el fallo del principal.** El respaldo se activó
   y nadie supo por qué. Ahora el principal se envuelve en un paso que registra
   el error antes de relanzarlo.
2. **El respaldo devolvía un objeto con ``.content``, no un mensaje.**
   ``StrOutputParser`` solo acepta texto o ``BaseMessage``, así que el camino
   de emergencia reventaba *dentro* del propio mecanismo de emergencia. El
   adaptador lo convierte a ``AIMessage``.

Los dobles de prueba no podían verlo: simulaban un modelo que devuelve
``AIMessage``, que es lo que devuelve Gemini y no lo que devolvía el wrapper.

Proyecto: ESSENTIAL — actividades prácticas de los Bloques 19 y 20.
"""

from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass
from typing import Protocol

from dotenv import load_dotenv
from langchain_core.messages import AIMessage
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
    sin_tildes,
)
from src.deepseek_wrapper import DeepSeekWrapper

logger = logging.getLogger(__name__)

# Contrato del material. Los dos números se usan en las plantillas y en la
# verificación: si cambian, cambian en un solo sitio.
SECCIONES_OBLIGATORIAS: tuple[str, ...] = ("objetivo", "conceptos", "ejercicio")
MAXIMO_PALABRAS = 200


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


def _normalizar_para_comparar(texto: str) -> str:
    """Minúsculas, sin tildes y con los espacios colapsados.

    Sirve para comparar dos cosas que deberían ser iguales aunque no lo
    parezcan: el nivel que llega como parámetro y el que responde el modelo, o
    una cita y el fragmento del documento del que salió. Sin colapsar los
    espacios, una cita correcta que cruza un salto de línea se marcaría como
    inventada.
    """

    return " ".join(sin_tildes(texto).split())


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


def _paso_nivel_fijo(nivel: str) -> Runnable:
    """Inyecta un nivel ya decidido, sin gastar una llamada al modelo.

    El nivel que llega de fuera se valida contra el mismo catálogo que la
    respuesta del modelo, así que un valor inesperado no deja la cadena sin
    destino: cae en el nivel por defecto y queda el aviso en el log.
    """

    normalizado = interpretar_nivel(nivel)

    if normalizado != _normalizar_para_comparar(nivel):
        logger.warning(
            "Nivel «%s» no reconocido; se usa «%s».",
            nivel,
            normalizado,
        )

    # El paso ignora la entrada a propósito: el nivel ya está decidido y lo
    # único que aporta es la clave que lee la bifurcación.
    return RunnablePassthrough.assign(nivel=lambda _: normalizado)


# --------------------------------------------------------------------------
# Las tres rutas: misma tarea, pedagogía distinta
# --------------------------------------------------------------------------

# Bloque común a las tres plantillas. Va en un solo sitio para que el contrato
# del material (secciones, citas, extensión) no se pueda desalinear entre
# rutas. Ojo con las llaves dobles: ``{{texto}}`` es lo que hace que la
# plantilla final tenga el marcador ``{texto}`` y no el valor del módulo.
_BLOQUE_FORMATO = (
    f"Formato obligatorio (tres secciones, en este orden):\n"
    f"**Objetivo:** una frase con lo que la persona sabrá hacer al terminar.\n"
    f"**Conceptos:** dos o tres, numerados. Cada uno lleva el nombre en "
    f"negrita, la explicación propia de este nivel y una cita literal del "
    f"documento entre comillas, en una línea aparte que empiece con «>». La "
    f"cita se copia palabra por palabra: no se traduce ni se parafrasea.\n"
    f"**Ejercicio:** uno, de una o dos frases.\n"
    f"Máximo {MAXIMO_PALABRAS} palabras, sin contar las citas.\n\n"
    f"Documento:\n"
    f"{{texto}}"
)

PLANTILLA_INICIAL = PromptTemplate.from_template(
    "Prepara material de capacitación para un público que parte de cero sobre "
    "el siguiente documento técnico.\n\n"
    "En cada concepto usa una analogía de la vida diaria y evita la jerga: si "
    "un término técnico es inevitable, defínelo en la misma frase. El "
    "ejercicio es guiado, paso a paso.\n\n"
    + _BLOQUE_FORMATO
)

PLANTILLA_INTERMEDIA = PromptTemplate.from_template(
    "Prepara material de capacitación para alguien que ya conoce el tema pero "
    "no la herramienta, a partir de este documento técnico.\n\n"
    "En cada concepto usa un ejemplo concreto de uso y relaciónalo con lo que "
    "la persona ya sabe del tema. El ejercicio es de aplicación, en una sola "
    "frase.\n\n"
    + _BLOQUE_FORMATO
)

PLANTILLA_AVANZADA = PromptTemplate.from_template(
    "Prepara material de capacitación para alguien que ya usa la herramienta, "
    "a partir de este documento técnico.\n\n"
    "En cada concepto explica la decisión de diseño que hay detrás y el "
    "trade-off real que esa decisión compra. El ejercicio es de extensión, en "
    "una sola frase.\n\n"
    + _BLOQUE_FORMATO
)


def construir_rutas(modelo: Runnable) -> dict[str, Runnable]:
    """Devuelve una ruta por nivel, cada una con su propia plantilla."""

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
    nivel: str | None = None,
) -> Runnable:
    """Manda el documento a la ruta que corresponde y devuelve el material.

    Args:
        modelo: modelo que clasifica (si hace falta) y redacta.
        rutas: rutas ya armadas. Se inyectan en las pruebas para no gastar
            llamadas y para verificar a cuál se fue.
        nivel: nivel del público, decidido por quien pide el material. Si
            viene informado **no se clasifica**: se ahorra una llamada y la
            ruta deja de depender del humor del modelo.

    ``RunnablePassthrough.assign`` conserva el ``texto`` de la entrada y le
    suma la clave ``nivel``: cada ruta sigue teniendo el documento original.
    """

    rutas = rutas or construir_rutas(modelo)

    if nivel is None:
        paso_nivel = RunnablePassthrough.assign(
            nivel=construir_paso_nivel(modelo)
        )
    else:
        paso_nivel = _paso_nivel_fijo(nivel)

    return paso_nivel | construir_bifurcacion(rutas)


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


class ModeloConInvoke(Protocol):
    """Cualquier objeto que exponga ``invoke`` sin ser un ``Runnable``."""

    def invoke(self, entrada: object) -> object:  # pragma: no cover
        ...


def adaptar_respaldo(modelo_de_respaldo: ModeloConInvoke) -> Runnable:
    """Adapta un objeto con ``invoke`` al protocolo de LangChain.

    Hacen falta dos cosas, no una:

    1. **Componer.** ``DeepSeekWrapper`` no es un ``Runnable`` ni un callable,
       así que sin envolverlo la cadena ni se arma y el respaldo queda como
       código muerto que parece funcionar.
    2. **Traducir la salida.** El wrapper devuelve un ``SimpleNamespace`` con
       ``.content``, y ``StrOutputParser`` solo acepta texto o ``BaseMessage``.
       Sin esto, el respaldo reventaba dentro del mecanismo de emergencia.
    """

    def invocar(entrada: object) -> AIMessage:
        respuesta = modelo_de_respaldo.invoke(entrada)
        contenido = getattr(respuesta, "content", respuesta)

        return AIMessage(content=contenido)

    return RunnableLambda(invocar)


def _con_registro(cadena: Runnable, etiqueta: str) -> Runnable:
    """Registra el fallo del principal **antes** de que el respaldo lo tape.

    ``with_fallbacks`` reintenta sin decir qué pasó. Sin este registro, un
    timeout o una cuota agotada quedan invisibles y solo se ve la respuesta del
    respaldo, como si el modelo principal nunca hubiera existido.

    Limitación: el ``invoke`` interno no hereda la configuración de ejecución,
    así que los callbacks y el streaming del principal no llegan. Aquí no se
    usan; el día que se usen, este envoltorio tiene que pasar a
    ``.with_listeners(on_error=...)``.
    """

    def ejecutar(entrada: dict[str, str]) -> object:
        try:
            return cadena.invoke(entrada)
        except ERRORES_DE_PROVEEDOR as error:
            logger.warning(
                "%s falló; se intenta el respaldo (%s: %s)",
                etiqueta,
                type(error).__name__,
                error,
            )
            raise

    return RunnableLambda(ejecutar)


def construir_cadena_segura(
    modelo: Runnable,
    modelo_respaldo: Runnable | None = None,
    nivel: str | None = None,
) -> Runnable:
    """Envuelve la cadena con respaldo y respuesta degradada.

    Los fallbacks se limitan a ``ERRORES_DE_PROVEEDOR``: un error de
    programación no se disfraza de caída del servicio. El nivel se propaga al
    respaldo para que las dos cadenas hagan exactamente lo mismo.
    """

    respaldos: list[Runnable] = []

    if modelo_respaldo is not None:
        respaldos.append(
            construir_cadena_con_enrutamiento(modelo_respaldo, nivel=nivel)
        )

    respaldos.append(RunnableLambda(_respuesta_generica))

    principal = _con_registro(
        construir_cadena_con_enrutamiento(modelo, nivel=nivel),
        etiqueta="modelo principal",
    )

    return principal.with_fallbacks(
        respaldos,
        exceptions_to_handle=ERRORES_DE_PROVEEDOR,
    )


# --------------------------------------------------------------------------
# Verificación del material: el contrato, medido
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class RevisionMaterial:
    """Resultado de comparar un material contra el contrato.

    Attributes:
        secciones_faltantes: etiquetas obligatorias que no aparecen.
        citas_invalidas: citas que no están, palabra por palabra, en el
            documento. Una cita inventada es la señal más barata de una
            alucinación: no requiere otro modelo para detectarla.
        palabras: palabras propias, sin contar las citas.
    """

    secciones_faltantes: tuple[str, ...]
    citas_invalidas: tuple[str, ...]
    palabras: int

    @property
    def aprueba(self) -> bool:
        """True si están las secciones, todas las citas son reales y cabe."""

        return (
            not self.secciones_faltantes
            and not self.citas_invalidas
            and self.palabras <= MAXIMO_PALABRAS
        )

    def resumen(self) -> str:
        """Una línea legible para el log."""

        partes = [f"{self.palabras} palabras"]

        if self.secciones_faltantes:
            partes.append(
                "faltan secciones: " + ", ".join(self.secciones_faltantes)
            )

        if self.citas_invalidas:
            partes.append(
                f"{len(self.citas_invalidas)} cita(s) que no están en el "
                "documento"
            )

        return " · ".join(partes)


def _etiquetas_presentes(material: str) -> set[str]:
    """Etiquetas de sección que aparecen al inicio de una línea.

    Se exige que estén al inicio y seguidas de dos puntos: buscar la palabra
    «objetivo» en cualquier parte del texto daría por buena una sección que el
    material solo menciona de pasada.
    """

    presentes: set[str] = set()

    for linea in material.splitlines():
        limpia = _normalizar_para_comparar(linea).lstrip("*# ").strip()

        for nombre in SECCIONES_OBLIGATORIAS:
            if limpia.startswith(f"{nombre}:"):
                presentes.add(nombre)

    return presentes


def _citas_del_material(material: str) -> list[str]:
    """Devuelve las citas: líneas que empiezan con ``>``, sin los adornos."""

    citas: list[str] = []

    for linea in material.splitlines():
        limpia = linea.strip()

        if limpia.startswith(">"):
            cita = limpia.lstrip(">").strip().strip("«»\"'`").strip()

            if cita:
                citas.append(cita)

    return citas


def _contar_palabras(material: str) -> int:
    """Cuenta las palabras propias: las citas son del documento, no nuestras."""

    propias = [
        linea
        for linea in material.splitlines()
        if not linea.strip().startswith(">")
    ]

    return len(" ".join(propias).split())


def verificar_material(material: str, documento: str) -> RevisionMaterial:
    """Compara el material contra el contrato y devuelve el informe.

    No lanza: un material imperfecto es un dato para el log, no una excepción.
    Si se quiere un portón, la decisión va en quien llama.
    """

    documento_normalizado = _normalizar_para_comparar(documento)

    citas = _citas_del_material(material)
    invalidas = tuple(
        cita
        for cita in citas
        if _normalizar_para_comparar(cita) not in documento_normalizado
    )

    presentes = _etiquetas_presentes(material)
    faltantes = tuple(
        nombre
        for nombre in SECCIONES_OBLIGATORIAS
        if nombre not in presentes
    )

    return RevisionMaterial(
        secciones_faltantes=faltantes,
        citas_invalidas=invalidas,
        palabras=_contar_palabras(material),
    )


# --------------------------------------------------------------------------
# Ejecución directa
# --------------------------------------------------------------------------

# Techo de la llamada al modelo. Con ``max_retries=0`` el peor caso es
# exactamente este número; el valor no busca dejar terminar al modelo (la ruta
# avanzada superó los 240 s en la medición), busca cortarlo temprano para que
# el respaldo tenga tiempo de responder.
TIEMPO_MAXIMO_POR_DEFECTO = 90.0
REINTENTOS_POR_DEFECTO = 0


def construir_modelo_gemini(
    clave: str,
    nombre: str | None = None,
) -> ChatGoogleGenerativeAI:
    """Arma el modelo con un techo de tiempo explícito.

    ``timeout`` en esta librería es *por intento*, así que el peor caso real es
    ``timeout × (1 + max_retries)``. Con ``max_retries=0`` el número que se lee
    en la configuración es el que de verdad se aplica.
    """

    return ChatGoogleGenerativeAI(
        model=nombre or os.getenv("GEMINI_MODEL", "gemini-3.8-flash"),
        google_api_key=clave,
        temperature=0,
        timeout=float(
            os.getenv("LLM_TIMEOUT_SECONDS", str(TIEMPO_MAXIMO_POR_DEFECTO))
        ),
        max_retries=int(
            os.getenv("LLM_MAX_REINTENTOS", str(REINTENTOS_POR_DEFECTO))
        ),
    )


def main() -> int:
    """Corre la cadena segura de extremo a extremo con el documento de ejemplo.

    Uso: ``python enrutamiento.py [nivel]``. Sin argumento, el nivel lo decide
    el modelo; con argumento se salta esa llamada.
    """

    configurar_registro()
    load_dotenv()

    nivel = sys.argv[1] if len(sys.argv) > 1 else None

    # Se registra aquí, donde el nivel entra de verdad. Dentro de la cadena se
    # registraría dos veces: la cadena segura arma la principal y el respaldo.
    if nivel is None:
        logger.info("Nivel: lo decide el modelo.")
    else:
        logger.info("Nivel solicitado: %s", nivel)

    clave_deepseek = os.getenv("DEEPSEEK_API_KEY")
    if not clave_deepseek:
        logger.error(
            "Falta DEEPSEEK_API_KEY: es el proveedor principal de esta cadena. "
            "Se espera en el .env del proyecto."
        )
        return 1

    # DeepSeek primero: es el proveedor que responde. ``DeepSeekWrapper`` no es
    # un ``Runnable``, así que pasa por el adaptador antes de entrar a la cadena.
    principal: Runnable = adaptar_respaldo(DeepSeekWrapper())

    respaldo: Runnable | None = None
    clave_gemini = os.getenv("GEMINI_API_KEY")

    if clave_gemini:
        respaldo = construir_modelo_gemini(clave_gemini)
    else:
        logger.info("Sin proveedor de respaldo: no hay GEMINI_API_KEY.")

    cadena = construir_cadena_segura(principal, respaldo, nivel)

    material = cadena.invoke({"texto": TEXTO_DE_EJEMPLO})

    logger.info("Resultado:\n%s", material)

    revision = verificar_material(material, TEXTO_DE_EJEMPLO)

    if revision.aprueba:
        logger.info("Revisión del material: cumple el contrato (%s).", revision.resumen())
    else:
        logger.warning("Revisión del material: fuera de contrato (%s).", revision.resumen())

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
